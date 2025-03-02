import logging
from task import *
import json
import hashlib
import os
import math
import concurrent.futures
import random
from server import *
from json_helper import *

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # this gets the directory where api.py is located
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "backend"))
SERVERS_JSON_PATH = os.path.join(PROJECT_DIR, "backend", "servers.json")

# approximate the Shapley value for each server by sampling random orders
# returns a dictionary mapping server.id to its estimated Shapley value
def compute_shapley_values(servers, n_samples=1000):
   
    n = len(servers)
    shapley = {server.id: 0.0 for server in servers}

    for _ in range(n_samples):
        perm = random.sample(servers, n)  # random ordering of servers
        coalition = []
        prev_value = 0.0
        for server in perm:
            coalition.append(server)
            new_value = value_function(coalition)
            marginal = new_value - prev_value
            shapley[server.id] += marginal
            prev_value = new_value

    # Average over samples
    for server_id in shapley:
        shapley[server_id] /= n_samples

    return shapley

# define the value of a coalition as the sum of (capacity - current_load)
def value_function(coalition):
    return sum(server.capacity - server.current_load for server in coalition)


class Auctioneer:
    def __init__(
        self,
        reserve_threshold: float = 25.0,
        decentralized: bool = False,
        servers_file="servers.json",
    ):
        self.reserve_threshold = reserve_threshold
        self.auction_log = []
        self.decentralized = decentralized
        self.servers_file = servers_file
        if decentralized:
            self.blockchain = []  # list of blocks for blockchain-style logging
    # adjust the reserve threshold based on recent auctions
    def update_reserve_threshold(self, round_num, recent_auctions):
        if recent_auctions:
            assigned_bids = [
                event["winning_bid"]
                for event in recent_auctions
                if event["status"]
                in ["assigned", "assigned_coalition", "assigned_above_threshold"]
            ]
            if assigned_bids:
                avg_bid = sum(assigned_bids) / len(assigned_bids)
                
                # adjust reserve threshold based on average bid, with a lower bound
                self.reserve_threshold = max(
                    avg_bid * 1.5, 10
                )  # prevent threshold going too low
        # otherwise leave unchanged
    # append a new block (containing the event) to the blockchain.
    def add_block(self, event):      
        prev_hash = self.blockchain[-1]["hash"] if self.blockchain else "0"
        block_data = json.dumps(event, sort_keys=True).encode()
        block_hash = hashlib.sha256(prev_hash.encode() + block_data).hexdigest()
        block = {"event": event, "prev_hash": prev_hash, "hash": block_hash}
        self.blockchain.append(block)

    # run auction - lowest bid wins, otherwise 
    def run_auction(self, task, servers, simulation=False):
        SERVER_FILE = SERVERS_JSON_PATH
        with open(SERVER_FILE, "r") as f:
            server_data = json.load(f)

            servers = [
                Server(s["id"], s["capacity"], s["sensitivity"]) for s in server_data
            ]

            print("Servers initialized:", servers)

        """
        
        if simulation: 
            SERVER_FILE = "servers.json"
            with open(SERVER_FILE, "r") as f:
                server_data = json.load(f)

                servers = [Server(s["id"], s["capacity"], s["sensitivity"]) for s in server_data]

                print("Servers initialized:", servers)
        else:
            # Initialize servers.
            servers = []
            for i in range(20):
                capacity = random.uniform(50, 100)
                sensitivity = random.uniform(0.5, 2.0)
                risk_aversion = random.uniform(0, 0.5)
                coalition_id = random.randint(0, 2 - 1)
                server = Server(
                    id=i,
                    capacity=capacity,
                    sensitivity=sensitivity,
                    target_load_ratio=0.5,
                    learning_rate=0.1,
                    learning_mode = "q-learning",
                    agent_params = {"epsilon": 0.1, "alpha": 0.1, "gamma": 0.9},
                    neural_bid_enabled=True,
                    risk_aversion=risk_aversion,
                    coalition_id=coalition_id)
                servers.append(server)

        """
        bids = {}
        for server in servers:
            bid = server.compute_bid(task)
            bids[server.id] = bid
            logging.debug(f"Server {server.id} bid {bid:.2f} for task {task.id}")

        # choose the server with the lowest bid
        winner_id = min(bids, key=bids.get)
        winner = next(s for s in servers if s.id == winner_id)
        sorted_bids = sorted(bids.items(), key=lambda x: x[1])
        second_price = sorted_bids[1][1] if len(sorted_bids) > 1 else sorted_bids[0][1]

        for i in range(100):
            print("Winner ID:", winner.id)

        # coalition logic if the winner is overloaded
        if (winner.current_load / winner.capacity) > 0.6:
            partners = [
                s
                for s in servers
                if s.coalition_id == winner.coalition_id
                and s.id != winner.id
                and (s.current_load / s.capacity) < 0.5
            ]
            if partners:
                partner = min(partners, key=lambda s: s.current_load / s.capacity)
                half_load = task.load / 2.0
                winner.assign_task(half_load)
                partner.assign_task(half_load)
                event = {
                    "task_id": task.id,
                    "winner": (winner.id, partner.id),
                    "winning_bid": bids[winner.id],
                    "second_price": second_price,
                    "bids": bids,
                    "status": "assigned_coalition",
                }
                self.auction_log.append(event)
                if self.decentralized:
                    self.add_block(event)
                logging.info(
                    f"Auction for task {task.id}: Coalition assignment between Server {winner.id} and Server {partner.id}."
                )
                return (winner, partner), bids[winner.id], second_price

        # Normal assignment
        if bids[winner.id] > self.reserve_threshold:
            status = "assigned_above_threshold"
            logging.info(
                f"Auction for task {task.id}: Winning bid {bids[winner.id]:.2f} exceeds reserve threshold ({self.reserve_threshold:.2f}). Assigning task to Server {winner.id} with penalty."
            )
        else:
            status = "assigned"
            logging.info(
                f"Auction for task {task.id}: Winner Server {winner.id} (bid {bids[winner.id]:.2f})."
            )

        event = {
            "task_id": task.id,
            "winner": winner.id,
            "winning_bid": bids[winner.id],
            "second_price": second_price,
            "bids": bids,
            "status": status,
        }
        self.auction_log.append(event)
        if self.decentralized:
            self.add_block(event)
        winner.assign_task(task.load)

        # check if winner is a tuple and extract the first element
        if isinstance(winner, tuple):
            winner = winner[0]  # assuming the first element is the actual Server object

        # ensure winner has an ID, otherwise assign one
        if not hasattr(winner, "id"):
            winner.id = id(winner)  # Assign a unique identifier based on memory address

        # if coalition was formed, log both winner and partner's ids
        if isinstance(winner, tuple):  # check if a tuple (coalition) was returned
            winner, partner = winner
            logging.info(
                f"Task assigned to Servers {winner.id} and {partner.id} (Winning bid: {bids[winner.id]:.2f}, Second price: {second_price:.2f})"
            )
        else:
            logging.info(
                f"Task assigned to Server {winner.id} (Winning bid: {bids[winner.id]:.2f}, Second price: {second_price:.2f})"
            )

        return winner, bids[winner.id], second_price

# approximate shapley values
def compute_shapley_values(servers, n_samples=1000):
    n = len(servers)
    shapley = {server.id: 0.0 for server in servers}

    for _ in range(n_samples):
        perm = random.sample(servers, n)  # random ordering of servers
        coalition = []
        prev_value = 0.0
        for server in perm:
            coalition.append(server)
            new_value = value_function(coalition)
            marginal = new_value - prev_value
            shapley[server.id] += marginal
            prev_value = new_value

    # average over samples
    for server_id in shapley:
        shapley[server_id] /= n_samples

    return shapley
