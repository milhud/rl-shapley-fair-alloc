import random

class Task:
    def __init__(self, id: int, load: float):
        self.id = id
        self.load = load
        # SLA: "gold", "silver", "bronze" (with different priorities)
        self.sla = random.choices(["gold", "silver", "bronze"], weights=[0.2, 0.5, 0.3])[0]
        # Complexity factor (affects processing difficulty)
        self.complexity = random.uniform(1.0, 3.0)
        # Latency requirement (in arbitrary time units)
        self.latency_requirement = random.uniform(1.0, 10.0)

def process_task(task, auctioneer, servers):
    """
    Helper function to process one task auction.
    Returns the auction result tuple.
    """
    return auctioneer.run_auction(task, servers)