import numpy as np
from typing import Dict, List, Any, Optional
from matcher import *

class ECSTable:
    """A columnar database table. Supports optional chunking for Bloom Filter skips."""
    def __init__(self, size: int, chunk_size: Optional[int] = 1024):
        self.size = size
        self.cols: Dict[str, np.ndarray] = {
            "id": np.arange(size, dtype=np.uint32),
            "mask": np.zeros(size, dtype=np.uint64) # Bitmask for tags/components
        }
        
        # Chunking metadata
        self.chunk_size = chunk_size
        self.num_chunks = (size + chunk_size - 1) // chunk_size
        self.chunk_summaries = np.zeros(self.num_chunks, dtype=np.uint64)

    def add_column(self, name: str, dtype: type, default_val=0):
        self.cols[name] = np.full(self.size, default_val, dtype=dtype)

    def set_tag(self, entity_id: int, tag_id: int):
        """Sets a tag and updates the chunk's Bloom Filter summary."""
        tag_bit = np.uint64(1) << np.uint64(tag_id % 64)
        self.cols["mask"][entity_id] |= tag_bit
        
        chunk_idx = entity_id // self.chunk_size
        self.chunk_summaries[chunk_idx] |= tag_bit

    def set_tags_batch(self, entity_ids: np.ndarray, tag_id: int):
        tag_bit = np.uint64(1) << np.uint64(tag_id % 64)
        self.cols["mask"][entity_ids] |= tag_bit
        chunk_indices = entity_ids // self.chunk_size
        np.bitwise_or.at(self.chunk_summaries, chunk_indices, tag_bit)

class SIMDEngine:
    """Executes Matcher ASTs against ECS Tables using NumPy vectorization."""
    
    @staticmethod
    def execute(table: ECSTable, matcher: Matcher) -> np.ndarray:
        optimized = sum_of_products(simplify(matcher))
        required_bits = SIMDEngine._extract_bits(optimized)
        
        # 1. No skip bits? Just one big slice.
        if required_bits == 0:
            return SIMDEngine._eval(table, optimized, slice(0, table.size))

        # 2. Vectorized identification of chunks to process
        # We only look at chunk_summaries (tiny array)
        active_chunk_indices = np.where((table.chunk_summaries & required_bits) == required_bits)[0]
        
        if len(active_chunk_indices) == 0:
            return np.zeros(table.size, dtype=bool)

        # 3. Process active chunks using Slices (Zero-Copy)
        full_mask = np.zeros(table.size, dtype=bool)
        for i in active_chunk_indices:
            start = i * table.chunk_size
            end = min(start + table.chunk_size, table.size)
            s = slice(start, end) # Slices are O(1) and zero-copy in NumPy
            full_mask[s] = SIMDEngine._eval(table, optimized, s)
                
        return full_mask

    @staticmethod
    def _eval(table: ECSTable, m: Matcher, s: slice) -> np.ndarray:
        """Recursive vectorized evaluator operating on a specific slice."""
        # Logical Operations
        if isinstance(m, And):    return SIMDEngine._eval(table, m.lhs, s) & SIMDEngine._eval(table, m.rhs, s)
        if isinstance(m, Or):     return SIMDEngine._eval(table, m.lhs, s) | SIMDEngine._eval(table, m.rhs, s)
        if isinstance(m, Not):    return ~SIMDEngine._eval(table, m.m, s)
        if isinstance(m, Always): return np.ones(s.stop - s.start, dtype=bool)
        if isinstance(m, Never):  return np.zeros(s.stop - s.start, dtype=bool)
        
        # Relational Operations (Leaves)
        col = table.cols[m.field][s]
        if isinstance(m, FieldEq): return col == m.value
        if isinstance(m, FieldNe): return col != m.value
        if isinstance(m, FieldLt): return col < m.value
        if isinstance(m, FieldGe): return col >= m.value
        if isinstance(m, FieldIn):
            # Only check IDs within the valid range of the bitmap
            # Bits outside the bitmap range are False by definition
            valid_range_mask = col < (len(m.bitmap) * 8)
            
            # Initialize result as False
            res = np.zeros_like(col, dtype=bool)
            
            # Only perform bit-math on valid indices to avoid IndexError
            valid_ids = col[valid_range_mask]
            byte_idx = valid_ids >> 3
            bit_mask = (1 << (valid_ids & 7)).astype(np.uint8)
            
            res[valid_range_mask] = (m.bitmap[byte_idx] & bit_mask) != 0
            return res
        
        raise ValueError(f"Unknown node: {m}")

    @staticmethod
    def _extract_bits(m: Matcher) -> np.uint64:
        """Recursively find required bits in AND chains."""
        if isinstance(m, FieldEq) and m.field == "mask":
            return np.uint64(m.value)
        if isinstance(m, And):
            return SIMDEngine._extract_bits(m.lhs) | SIMDEngine._extract_bits(m.rhs)
        return np.uint64(0)
    
    @staticmethod
    def semi_join(target_table: ECSTable, target_query: Matcher, 
                             local_fk_field: str) -> Matcher:
        """Dynamically produces a FieldIn constraint from a sub-query."""
        target_mask = SIMDEngine.execute(target_table, target_query)
        bitmap = np.packbits(target_mask, bitorder='little')
        return FieldIn(local_fk_field, bitmap)
    
    @staticmethod
    def semi_naive_join(table: ECSTable, initial_query: Matcher, 
                        fk_field: str, id_field: str = "id") -> Matcher:
        """
        Semi-Naive recursion: Only joins the NEWLY found entities in each step.
        """
        delta_mask = SIMDEngine.execute(table, initial_query)
        total_mask = delta_mask.copy()
        
        while np.any(delta_mask):
            count = np.count_nonzero(delta_mask)
            frontier_query = None

            if count < 32:
                # If the frontier is tiny, a set of FieldEq is faster than a Bitmap
                frontier_ids = np.where(delta_mask)[0]
                frontier_query = Or.from_list([FieldEq(fk_field, fid) for fid in frontier_ids])
            else:
                # If the frontier is large, stick to the Bitmap
                delta_bitmap = np.packbits(delta_mask, bitorder='little')
                frontier_query = FieldIn(fk_field, delta_bitmap)
            
            # 3. Find neighbors of the frontier
            reachable_from_delta = SIMDEngine.execute(table, frontier_query)
            
            # 4. CRITICAL: New Delta = (Reachable) AND NOT (Already Found)
            # This prevents re-processing the same entities forever
            delta_mask = reachable_from_delta & ~total_mask
            total_mask |= delta_mask
        return FieldIn(id_field, np.packbits(total_mask, bitorder='little'))

    @staticmethod
    def aggregate(table: ECSTable, query: Matcher, column: str, op: str = "sum"):
        # 1. Get the mask for the entities we care about
        mask = SIMDEngine.execute(table, query)
        
        # 2. Extract the active data (Zero-copy view where possible)
        data = table.cols[column][mask]
        
        # 3. Perform the reduction
        if op == "sum":   return np.sum(data)
        if op == "max":   return np.max(data)
        if op == "min":   return np.min(data)
        if op == "mean":  return np.mean(data)
        if op == "count": return len(data)
        
        raise ValueError(f"Unsupported operator: {op}")