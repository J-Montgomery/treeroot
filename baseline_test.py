import time
import random
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Player:
    id: int
    team: int
    health: float

@dataclass
class Pet:
    id: int
    owner_id: int
    species_id: int

def run_baseline_comparison(num_entities: int):
    print(f"--- Running Baseline Comparison ({num_entities} entities) ---")
    
    # 1. Setup Data
    # In OO baseline, data is stored as a list of objects (Row-based)
    players = [Player(i, random.randint(1, 3), random.uniform(0, 100)) 
               for i in range(num_entities)]
    
    pets = [Pet(i, random.randint(0, num_entities-1), random.randint(1, 10)) 
            for i in range(num_entities)]

    # 2. Define the Query
    # Target: Pets of Species 5 whose owners are Team 2 with < 20 health.
    
    start_time = time.time()

    # --- Step A: Find valid owners (The Join Prep) ---
    # This involves iterating over all players and checking conditions
    valid_owner_ids = {p.id for p in players if p.team == 2 and p.health < 20.0}

    # --- Step B: Filter Pets (The Join Execution) ---
    # This involves iterating over all pets and performing a hash-set lookup
    results = [pt.id for pt in pets 
               if pt.species_id == 5 and pt.owner_id in valid_owner_ids]

    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Baseline (Object/Loop): Found {len(results)} pets in {duration:.4f} seconds.")
    return duration

if __name__ == "__main__":
    # We use a smaller number for baseline if needed, but 2M is fine for comparison
    baseline_time = run_baseline_comparison(4_000_000)