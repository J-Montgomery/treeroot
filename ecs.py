import numpy as np
from typing import Dict
from matcher import *

class ECSTable:
    """A columnar database table representing entities and their components."""
    def __init__(self, size: int):
        self.size = size
        self.cols: Dict[str, np.ndarray] = {
            "id": np.arange(size, dtype=np.uint32),
            "mask": np.zeros(size, dtype=np.uint64) # Bitmask for components
        }
        
    def add_column(self, name: str, dtype: type, default_val=0):
        self.cols[name] = np.full(self.size, default_val, dtype=dtype)

class SIMDEngine:
    """Executes Matcher ASTs against ECS Tables using NumPy vectorization."""
    
    @staticmethod
    def execute(table: ECSTable, matcher: Matcher) -> np.ndarray:
        """Returns a boolean mask of entities that match the simplified query."""
        # 1. Compile-Time Algebra Optimization
        optimized = sum_of_products(simplify(matcher))
        
        # 2. SIMD Evaluation
        return SIMDEngine._eval(table, optimized)

    @staticmethod
    def _eval(table: ECSTable, m: Matcher) -> np.ndarray:
        # Vectorized Logical Operations
        if isinstance(m, Always):  return np.ones(table.size, dtype=bool)
        if isinstance(m, Never):   return np.zeros(table.size, dtype=bool)
        if isinstance(m, And):     return SIMDEngine._eval(table, m.lhs) & SIMDEngine._eval(table, m.rhs)
        if isinstance(m, Or):      return SIMDEngine._eval(table, m.lhs) | SIMDEngine._eval(table, m.rhs)
        if isinstance(m, Not):     return ~SIMDEngine._eval(table, m.m)
        
        # Vectorized Relational Operations
        if isinstance(m, FieldEq): return table.cols[m.field] == m.value
        if isinstance(m, FieldNe): return table.cols[m.field] != m.value
        if isinstance(m, FieldLt): return table.cols[m.field] < m.value
        if isinstance(m, FieldGe): return table.cols[m.field] >= m.value
        if isinstance(m, FieldIn):
            fks = table.cols[m.field]
            byte_idx = fks >> 3
            bit_mask = (1 << (fks & 7)).astype(np.uint8)
            return (m.bitmap[byte_idx] & bit_mask) != 0
        
        raise ValueError(f"Unknown execution node: {m}")
    
    @staticmethod
    def semi_join(target_table: ECSTable, target_query: Matcher, 
                             local_fk_field: str) -> Matcher:
        target_mask = SIMDEngine.execute(target_table, target_query)
        bitmap = np.packbits(target_mask, bitorder='little')
        return FieldIn(local_fk_field, bitmap)

class ChunkedECSTable(ECSTable):
    def __init__(self, size: int, chunk_size: int = 1024):
        super().__init__(size)
        self.chunk_size = chunk_size
        self.num_chunks = (size + chunk_size - 1) // chunk_size
        
        # A Bloom Filter summary per chunk. 
        # Each bit represents a Tag or Component ID present in this chunk.
        self.chunk_summaries = np.zeros(self.num_chunks, dtype=np.uint64)

    def set_tag(self, entity_id: int, tag_id: int):
        """Sets a tag and updates the chunk's Bloom Filter summary."""
        chunk_idx = entity_id // self.chunk_size
        
        # Force the bit to uint64 to match the column's dtype
        tag_bit = np.uint64(1) << np.uint64(tag_id % 64)
        
        self.cols["mask"][entity_id] |= tag_bit
        self.chunk_summaries[chunk_idx] |= tag_bit

class OptimizedSIMDEngine(SIMDEngine):
    @staticmethod
    def _extract_required_bits(m: Matcher) -> np.uint64:
        """Extracts bitmasks and ensures they are uint64 for NumPy compatibility."""
        if isinstance(m, FieldEq) and m.field == "mask":
            return np.uint64(m.value) # Force casting here
        if isinstance(m, And):
            return np.uint64(OptimizedSIMDEngine._extract_required_bits(m.lhs) | \
                             OptimizedSIMDEngine._extract_required_bits(m.rhs))
        return np.uint64(0)

    @staticmethod
    def execute(table: ChunkedECSTable, matcher: Matcher) -> np.ndarray:
        optimized = sum_of_products(simplify(matcher))
        
        # Ensure the filter mask is the correct type before the loop
        required_bits = np.uint64(OptimizedSIMDEngine._extract_required_bits(optimized))
        
        full_mask = np.zeros(table.size, dtype=bool)
        
        # Optimization: If no bits are required, skip the bloom check entirely
        if required_bits == 0:
             return SIMDEngine._eval(table, optimized)

        for i in range(table.num_chunks):
            # Now both sides are guaranteed to be uint64
            if (table.chunk_summaries[i] & required_bits) == required_bits:
                start = i * table.chunk_size
                end = min(start + table.chunk_size, table.size)
                chunk_slice = slice(start, end)
                full_mask[chunk_slice] = OptimizedSIMDEngine._eval_slice(table, optimized, chunk_slice)
                
        return full_mask

    @staticmethod
    def _eval_slice(table: ChunkedECSTable, m: Matcher, s: slice) -> np.ndarray:
        """Corrected slice evaluator with full recursive support for logical nodes."""
        
        # 1. Handle Recursive Logical Operations
        # These must call _eval_slice again to stay within the chunk bounds
        if isinstance(m, And):
            return OptimizedSIMDEngine._eval_slice(table, m.lhs, s) & \
                OptimizedSIMDEngine._eval_slice(table, m.rhs, s)
        
        if isinstance(m, Or):
            return OptimizedSIMDEngine._eval_slice(table, m.lhs, s) | \
                OptimizedSIMDEngine._eval_slice(table, m.rhs, s)
        
        if isinstance(m, Not):
            return ~OptimizedSIMDEngine._eval_slice(table, m.m, s)

        if isinstance(m, Always): 
            return np.ones(s.stop - s.start, dtype=bool)
        
        if isinstance(m, Never):  
            return np.zeros(s.stop - s.start, dtype=bool)

        # 2. Handle Leaf Relational Operations
        # We slice the column first, then perform the vectorized comparison
        col = table.cols[m.field][s]
        
        if isinstance(m, FieldEq): return col == m.value
        if isinstance(m, FieldNe): return col != m.value
        if isinstance(m, FieldLt): return col < m.value
        if isinstance(m, FieldGe): return col >= m.value
        
        if isinstance(m, FieldIn):
            # fks is now just the chunk's portion of the foreign keys
            byte_idx = col >> 3
            bit_mask = (1 << (col & 7)).astype(np.uint8)
            return (m.bitmap[byte_idx] & bit_mask) != 0

        raise ValueError(f"Unknown execution node: {m}")