import time
import numpy as np
from matcher import *
from ecs import ECSTable, SIMDEngine, ChunkedECSTable, OptimizedSIMDEngine

def test_algebraic_simplifications():
    print("--- Testing Boolean Algebra Compiler ---")
    x_eq_5 = FieldEq("x", 5)
    x_lt_10 = FieldLt("x", 10)
    
    # 1. Identity: X & X -> X
    assert simplify(x_eq_5 & x_eq_5) == x_eq_5
    
    # 2. Subsumption: (X == 5) implies (X < 10), so (X==5) & (X<10) -> (X==5)
    res1 = simplify(x_eq_5 & x_lt_10)
    print(f"Subsumption (And): {res1}")
    assert res1 == x_eq_5
    
    # 3. Subsumption (Or): (X==5) | (X<10) -> (X < 10)
    res2 = simplify(x_eq_5 | x_lt_10)
    print(f"Subsumption (Or) : {res2}")
    assert res2 == x_lt_10
    
    # 4. Contradiction: (X == 5) & (X != 5) -> Never
    res3 = simplify(x_eq_5 & negate(x_eq_5))
    print(f"Contradiction    : {res3}")
    assert isinstance(res3, Never)

def test_ecs_with_joins():
    print("\n--- Testing SIMD ECS & Joins ---")
    NUM_ENTITIES = 2_000_000
    
    # Setup Players Table
    players = ECSTable(NUM_ENTITIES)
    players.add_column("team", np.int32)
    players.add_column("health", np.float32)
    
    # Populate random data
    players.cols["team"] = np.random.randint(1, 4, NUM_ENTITIES)
    players.cols["health"] = np.random.uniform(0, 100, NUM_ENTITIES)
    
    # Setup Pets Table (Relates to Players)
    pets = ECSTable(NUM_ENTITIES)
    pets.add_column("owner_id", np.uint32)
    pets.add_column("species_id", np.int32)
    
    pets.cols["owner_id"] = np.random.randint(0, NUM_ENTITIES, NUM_ENTITIES)
    pets.cols["species_id"] = np.random.randint(1, 10, NUM_ENTITIES)

    # --- THE QUERY ---
    # Find all pets of Species 5 that are owned by a Player on Team 2 with < 20 health.
    
    start_time = time.time()
    
    # 1. Define target query for Players
    player_query = FieldEq("team", 2) & FieldLt("health", 20.0)
    
    # 2. Execute Semi-Join to compile constraints dynamically
    join_matcher = SIMDEngine.semi_join(
        target_table=players, 
        target_query=player_query, 
        local_fk_field="owner_id"
    )
    
    # 3. Combine with local constraints
    pet_query = FieldEq("species_id", 5) & join_matcher
    
    # 4. Execute final SIMD evaluation on Pets
    matching_pets_mask = SIMDEngine.execute(pets, pet_query)
    matching_pet_ids = pets.cols["id"][matching_pets_mask]
    
    end_time = time.time()
    
    print(f"Found {len(matching_pet_ids)} matching pets.")
    print(f"Execution time for {NUM_ENTITIES * 2} records: {end_time - start_time:.4f} seconds.")

def test_sparse_performance():
    NUM_ENTITIES = 2_000_000
    
    # Setup Players Table
    players = ECSTable(NUM_ENTITIES)
    players.add_column("team", np.int32)
    players.add_column("health", np.float32)

    # 1. Create a Presence Mask (1 = has component, 0 = doesn't)
    # Only 10% of entities have the 'Health' component
    has_health = np.random.choice([0, 1], size=NUM_ENTITIES, p=[0.9, 0.1])
    
    # 2. Your SIMD Logic likely does this:
    # (has_health) & (health < 20) & (team == 2)
    start = time.time()
    
    # Standard NumPy/SIMD bitwise 'AND' is effectively free
    mask = (has_health == 1) & (players.cols["health"] < 20) & (players.cols["team"] == 2)
    
    print(f"Sparse Query Time: {time.time() - start:.4f}s")

def test_logic_bomb():
    print("\n--- Testing Logic Bomb (OR-Chain) ---")
    NUM_ENTITIES = 2_000_000

    # A redundant mess of logic:
    # (Health < 20) OR (Health < 50) OR (Health < 10) OR (Health < 100)
    # This SHOULD simplify to just: FieldLt("health", 100.0)
    players = ECSTable(NUM_ENTITIES)
    players.add_column("team", np.int32)
    players.add_column("health", np.float32)

    redundant_query = (
        FieldLt("health", 20.0) | 
        FieldLt("health", 50.0) | 
        FieldLt("health", 10.0) | 
        FieldLt("health", 100.0)
    )
    
    start_compile = time.time()
    optimized_query = simplify(redundant_query)
    compile_time = time.time() - start_compile
    
    print(f"Original Query: 4 OR conditions")
    print(f"Optimized to  : {optimized_query}")
    print(f"Compile logic time: {compile_time:.6f}s")
    
    # Execute on 2M records
    start_exec = time.time()
    mask = SIMDEngine.execute(players, optimized_query)
    exec_time = time.time() - start_exec
    
    print(f"Execution time for 2,000,000 records: {exec_time:.4f}s")

def test_triple_join():
    print("\n--- Testing Triple-Join (Players -> Pets -> Items) ---")
    NUM_ENTITIES = 2_000_000
    
    # 1. Players Table
    players = ECSTable(NUM_ENTITIES)
    players.add_column("team", np.int32)
    players.cols["team"] = np.random.randint(1, 4, NUM_ENTITIES)
    
    # 2. Pets Table (Linked to Players)
    pets = ECSTable(NUM_ENTITIES)
    pets.add_column("owner_id", np.uint32)
    pets.cols["owner_id"] = np.random.randint(0, NUM_ENTITIES, NUM_ENTITIES)
    
    # 3. Items Table (Linked to Pets)
    items = ECSTable(NUM_ENTITIES)
    items.add_column("pet_id", np.uint32)
    items.add_column("rarity", np.int32)
    items.cols["pet_id"] = np.random.randint(0, NUM_ENTITIES, NUM_ENTITIES)
    items.cols["rarity"] = np.random.randint(1, 100, NUM_ENTITIES)

    # --- THE TRIPLE JOIN QUERY ---
    # Find all 'Legendary' Items (Rarity > 90) belonging to Pets 
    # whose Owners are on Team 2.
    
    start_time = time.time()
    
    # Path: Players (Team 2) -> Pets (Semi-Join) -> Items (Semi-Join)
    
    # Step A: Filter Players on Team 2
    player_query = FieldEq("team", 2)
    
    # Step B: Join Players to Pets
    pet_join_matcher = SIMDEngine.semi_join(
        target_table=players,
        target_query=player_query,
        local_fk_field="owner_id"
    )
    
    # Step C: Join Pets to Items
    item_join_matcher = SIMDEngine.semi_join(
        target_table=pets,
        target_query=pet_join_matcher, # Passing the previous join result!
        local_fk_field="pet_id"
    )
    
    # Step D: Final local filter on Items
    final_query = FieldGe("rarity", 91) & item_join_matcher
    
    results = SIMDEngine.execute(items, final_query)
    
    end_time = time.time()
    
    print(f"Found {np.sum(results)} legendary items for Team 2.")
    print(f"Triple-Join execution time: {end_time - start_time:.4f} seconds.")


def test_negative_join():
    print("\n--- Testing Negative-Join (Not Team 2) ---")
    NUM_ENTITIES = 2_000_000
    
    # 1. Players Table
    players = ECSTable(NUM_ENTITIES)
    players.add_column("team", np.int32)
    players.cols["team"] = np.random.randint(1, 4, NUM_ENTITIES)
    
    # 2. Pets Table (Linked to Players)
    pets = ECSTable(NUM_ENTITIES)
    pets.add_column("owner_id", np.uint32)
    pets.cols["owner_id"] = np.random.randint(0, NUM_ENTITIES, NUM_ENTITIES)
    
    items = ECSTable(NUM_ENTITIES)
    items.add_column("pet_id", np.uint32)
    items.add_column("rarity", np.int32)
    items.cols["pet_id"] = np.random.randint(0, NUM_ENTITIES, NUM_ENTITIES)
    items.cols["rarity"] = np.random.randint(1, 100, NUM_ENTITIES)

    start_time = time.time()
    
    # 1. Negative Query: Everyone EXCEPT Team 2
    # This can be represented as FieldNe("team", 2) 
    # OR negate(FieldEq("team", 2))
    player_query = negate(FieldEq("team", 2))
    
    # 2. Propagate the 'Negative' status through the Joins
    # The Semi-join doesn't care if the mask represents 'In' or 'Out'
    # It just maps 'True' bits across tables.
    pet_join_matcher = SIMDEngine.semi_join(
        target_table=players,
        target_query=player_query,
        local_fk_field="owner_id"
    )
    
    item_join_matcher = SIMDEngine.semi_join(
        target_table=pets,
        target_query=pet_join_matcher,
        local_fk_field="pet_id"
    )
    
    # 3. Combine and Simplify
    final_query = simplify(FieldGe("rarity", 91) & item_join_matcher)
    
    results = OptimizedSIMDEngine.execute(items, final_query)
    end_time = time.time()
    
    print(f"Found {np.sum(results)} items for non-Team 2 players.")
    print(f"Negative-Join execution time: {end_time - start_time:.4f} seconds.")

def test_hierarchical_tags_with_bloom():
    print("\n--- Testing Hierarchical Tags (Bloom Filter Skip) ---")
    NUM_ENTITIES = 2_000_000
    CHUNK_SIZE = 1024
    
    # 1. Setup Table
    # mask will store our hierarchical bits
    items = ChunkedECSTable(NUM_ENTITIES, chunk_size=CHUNK_SIZE)
    items.add_column("durability", np.int32, default_val=100)
    
    # 2. Define Tag IDs (Simulating Hierarchy)
    # Item = Bit 0, Weapon = Bit 1, Sword = Bit 2
    TAG_ITEM   = 1 << 0  # 001
    TAG_WEAPON = 1 << 1  # 010
    TAG_SWORD  = 1 << 2  # 100
    
    # We want a "Legendary Sword" to have all three bits: 111 (Value 7)
    LEGENDARY_SWORD_MASK = TAG_ITEM | TAG_WEAPON | TAG_SWORD
    
    # 3. Populate Data Sparsely
    # Let's place Legendary Swords only in every 100th chunk to test skipping.
    print(f"Populating {NUM_ENTITIES} entities...")
    for chunk_id in range(0, items.num_chunks, 100):
        # Only put 5 swords at the start of every 100th chunk
        start_idx = chunk_id * CHUNK_SIZE
        for i in range(5):
            idx = start_idx + i
            if idx < NUM_ENTITIES:
                items.set_tag(idx, 0) # Item
                items.set_tag(idx, 1) # Weapon
                items.set_tag(idx, 2) # Sword
    
    # 4. The Query
    # Find all entities that are "Swords" (must have bit 2) 
    # This matches our Bloom Filter 'required_bits' logic.
    query = FieldEq("mask", LEGENDARY_SWORD_MASK) & FieldLt("durability", 150)
    
    # 5. Execute with Timing
    start_time = time.time()
    results_mask = OptimizedSIMDEngine.execute(items, query)
    end_time = time.time()
    
    num_found = np.sum(results_mask)
    chunks_processed = items.num_chunks 
    # In a real impl, you'd increment a counter in the 'if' block to see skips
    
    print(f"Found {num_found} Legendary Swords.")
    print(f"Bloom Filter Execution Time: {end_time - start_time:.6f}s")
    
    # 6. Compare to Raw Scan (for PoC validation)
    # This simulates what happens if the Bloom Filter fails/is disabled
    start_raw = time.time()
    raw_mask = (items.cols["mask"] & TAG_SWORD == TAG_SWORD) & (items.cols["durability"] < 150)
    end_raw = time.time()
    
    print(f"Raw Vectorized Scan Time:   {end_raw - start_raw:.6f}s")
    print(f"Speedup Factor: { (end_raw - start_raw) / (end_time - start_time):.2f}x")

if __name__ == "__main__":
    test_algebraic_simplifications()
    test_ecs_with_joins()
    test_sparse_performance()
    test_logic_bomb()
    test_triple_join()
    test_negative_join()
    test_hierarchical_tags_with_bloom()