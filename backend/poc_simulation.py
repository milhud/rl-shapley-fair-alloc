#!/usr/bin/env python3
"""
Mega Simulation for Advanced Network Load Balancing

Features incorporated:
1. Q–learning / Multi–armed bandit for updating bidding sensitivity.
2. Dynamic bid adjustment using a dummy neural network predictor.
3. Risk–sensitive bidding: bids are adjusted by a risk–aversion factor.
4. Extended task characteristics: SLA level, complexity, latency.
5. Non–stationary task arrival (sinusoidally varying average).
6. Adaptive reserve price in the auctioneer.
7. Coalition formation: overloaded servers in the same coalition may offload half a task.
8. Fairness and social welfare metrics.
9. Decentralized (blockchain–style) auction logging.

Note: This is a prototype. In practice, you may wish to replace the dummy neural predictor with a true NN,
refine learning updates, and modularize the code.
"""

import random
import logging
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import hashlib
import concurrent.futures
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Configure logging (set to DEBUG to see detailed internal state updates)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def process_task(task, auctioneer, servers):
    """
    Helper function to process one task auction.
    Returns the auction result tuple.
    """
    return auctioneer.run_auction(task, servers)

###############################################################################
# 1. LearningAgent: Q–learning or Multi–armed Bandit agent for discrete decisions.
###############################################################################
class LearningAgent:
    def __init__(self, mode="q-learning", actions=["increase", "decrease", "none"],
                 epsilon=0.1, alpha=0.1, gamma=0.9):
        self.mode = mode
        self.actions = actions
        self.epsilon = epsilon  # exploration probability
        self.alpha = alpha      # learning rate for Q–learning
        self.gamma = gamma      # discount factor for Q–learning

        if self.mode == "q-learning":
            self.Q = {}  # Q–table: mapping state -> {action: value}
        elif self.mode == "bandit":
            # For bandit mode, we keep estimated values for each action (state is ignored)
            self.values = {action: 0.0 for action in actions}
            self.counts = {action: 0 for action in actions}

    def select_action(self, state=None):
        if self.mode == "q-learning":
            if state not in self.Q:
                self.Q[state] = {action: 0.0 for action in self.actions}
            if random.random() < self.epsilon:
                action = random.choice(self.actions)
            else:
                action = max(self.Q[state], key=self.Q[state].get)
            return action
        elif self.mode == "bandit":
            if random.random() < self.epsilon:
                return random.choice(self.actions)
            else:
                return max(self.values, key=self.values.get)
        else:
            return "none"

    def update(self, state, action, reward, next_state=None):
        if self.mode == "q-learning":
            if state not in self.Q:
                self.Q[state] = {a: 0.0 for a in self.actions}
            if next_state is None or next_state not in self.Q:
                next_max = 0.0
            else:
                next_max = max(self.Q[next_state].values())
            self.Q[state][action] += self.alpha * (reward + self.gamma * next_max - self.Q[state][action])
        elif self.mode == "bandit":
            self.counts[action] += 1
            n = self.counts[action]
            self.values[action] += (1.0 / n) * (reward - self.values[action])

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, input_dim, output_dim, gamma=0.99, epsilon=0.1, epsilon_decay=0.995, lr=0.001):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = 32
        self.replay_buffer = deque(maxlen=10000)
        self.q_network = DQN(input_dim, output_dim)
        self.target_network = DQN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_action(self, state):
        # state should be a torch tensor of shape [1, input_dim]
        if random.random() < self.epsilon:
            return random.randrange(self.q_network.fc3.out_features)
        else:
            with torch.no_grad():
                q_vals = self.q_network(state)
            return torch.argmax(q_vals).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.cat(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_network(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        self.update_target_network()

###############################################################################
# 2. NeuralBidPredictor: A dummy predictor that “learns” a bid adjustment.
###############################################################################
class NeuralBidPredictor:
    def __init__(self):
        # Dummy parameter: a single weight that scales the sum of input features.
        self.weight = random.uniform(0.8, 1.2)

    def predict_bid(self, features):
        # Features is a list of numbers (e.g., [load_ratio, task_load, sensitivity, risk_aversion]).
        return self.weight * sum(features)

    def update_model(self, features, target):
        # A dummy online update: adjust weight slightly to reduce prediction error.
        prediction = self.predict_bid(features)
        error = target - prediction
        self.weight += 0.01 * error

###############################################################################
# 3. Server: Represents a server with advanced bidding and learning.
###############################################################################
class Server:
    def __init__(self, id: int, capacity: float, sensitivity: float,
                 target_load_ratio: float = 0.5, learning_rate: float = 0.1,
                 learning_mode: str = "continuous",  # Options: "continuous", "q-learning", "bandit"
                 agent_params: dict = None,
                 neural_bid_enabled: bool = False,
                 risk_aversion: float = 0.1,
                 coalition_id: int = 0):
        self.id = id
        self.capacity = capacity
        self.sensitivity = sensitivity
        self.current_load = 0.0
        self.target_load_ratio = target_load_ratio
        self.learning_rate = learning_rate

        self.risk_aversion = risk_aversion  # Factor that modulates risk sensitivity.
        self.coalition_id = coalition_id    # ID indicating coalition membership.

        self.bid_history = []
        self.task_history = []
        self.sensitivity_history = [sensitivity]

        # Learning mode for adjusting sensitivity.
        self.learning_mode = learning_mode
        self.learning_agent = LearningAgent(mode=learning_mode, **(agent_params or {}))
        self.last_state = None
        self.last_action = None
        self.last_error = None

        # Neural bid predictor (dummy NN) for bid adjustment.
        self.neural_bid_enabled = neural_bid_enabled
        if self.neural_bid_enabled:
            # For example, 4 input features and 3 actions
            self.dqn_agent = DQNAgent(input_dim=4, output_dim=3)
        else:
            self.dqn_agent = None

    def compute_bid(self, task) -> float:
        """
        Compute a bid for a given task.
        Bid is based on:
        - current load ratio (current_load/capacity) (clamped to 1.0)
        - task load
        - server sensitivity
        - a risk adjustment factor that only increases if the server is overloaded
        - a small random noise.
        Optionally, a neural predictor may add an adjustment based on features.
        """
        # Calculate load_ratio and clamp it to a maximum of 1.0
        load_ratio = self.current_load / self.capacity
        load_ratio = min(load_ratio, 1.0)

        # Compute a base bid using sensitivity and task load
        base_bid = self.sensitivity * load_ratio + task.load

        # Only add risk adjustment if load_ratio exceeds target_load_ratio
        risk_adjustment = 1 + self.risk_aversion * max(0, (load_ratio - self.target_load_ratio))
        base_bid *= risk_adjustment

        # Add a small random noise
        noise = random.uniform(0, 0.1)

        # Use neural bid adjustment if enabled
        if self.neural_bid_enabled and self.dqn_agent:
            # Create state tensor: [load_ratio, task.load, sensitivity, risk_aversion]
            state = torch.tensor([[load_ratio, task.load, self.sensitivity, self.risk_aversion]], dtype=torch.float32)
            action = self.dqn_agent.get_action(state)
            # Map action to a bid adjustment factor:
            if action == 0:
                adjustment = self.learning_rate  # increase bid adjustment
            elif action == 1:
                adjustment = -self.learning_rate  # decrease bid adjustment
            else:
                adjustment = 0.0  # no change
            bid = base_bid * (1 + adjustment) + noise
        else:
            bid = base_bid + noise

        # Dampening the capacity multiplier: reduce from (1 + capacity/100) to (1 + capacity/200)
        bid *= (1 + (self.capacity / 200))

        # Optionally, clamp the bid to a maximum value to avoid runaway bids.
        bid = min(bid, 1000.0)

        self.bid_history.append(bid)
        return bid

    def assign_task(self, task_load: float):
        """
        Increase current load by the task load.
        """
        self.current_load += task_load
        self.task_history.append(task_load)
        logging.info(f"Server {self.id} assigned task load {task_load:.2f}, new load: {self.current_load:.2f}")

    def process_load(self, processing_rate: float):
        """
        Simulate processing by reducing load.
        """
        old_load = self.current_load
        self.current_load = max(0, self.current_load - processing_rate)
        if old_load != self.current_load:
            logging.debug(f"Server {self.id} processed load: {old_load:.2f} -> {self.current_load:.2f}")

    def update_strategy(self):
        """
        Update sensitivity.
        In continuous mode: sensitivity *= (1 + learning_rate * (load_ratio - target_load_ratio))
        In q-learning or bandit mode: discretize the load ratio into a state and select an action.
        """
        load_ratio = self.current_load / self.capacity
        current_state = int(load_ratio * 10)  # discretized state: 0 to 10
        current_error = abs(load_ratio - self.target_load_ratio)

        if self.learning_mode in ["q-learning", "bandit"]:
            if self.last_state is not None:
                reward = self.last_error - current_error  # positive if error decreased
                self.learning_agent.update(self.last_state, self.last_action, reward, current_state)
                logging.debug(f"Server {self.id} learning update: state {self.last_state}, action {self.last_action}, reward {reward:.3f}")

            action = self.learning_agent.select_action(current_state)  # This is where action is selected

            old_sensitivity = self.sensitivity
            if action == "increase":
                self.sensitivity *= (1 + self.learning_rate)
            elif action == "decrease":
                self.sensitivity *= (1 - self.learning_rate)
            elif action == "none":
                pass
            self.sensitivity = max(self.sensitivity, 0.1)

            self.sensitivity_history.append(self.sensitivity)
            logging.debug(f"Server {self.id} applied action '{action}': sensitivity {old_sensitivity:.3f} -> {self.sensitivity:.3f}")

            # Store last state, action, and error for the next update
            self.last_state = current_state
            self.last_action = action
            self.last_error = current_error
        else:
            delta = load_ratio - self.target_load_ratio
            old_sensitivity = self.sensitivity
            self.sensitivity *= (1 + self.learning_rate * delta)
            self.sensitivity = max(self.sensitivity, 0.1)
            self.sensitivity_history.append(self.sensitivity)
            logging.debug(f"Server {self.id} updated sensitivity continuously: {old_sensitivity:.3f} -> {self.sensitivity:.3f}")

###############################################################################
# 4. Task: Represents a task with additional characteristics.
###############################################################################
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

###############################################################################
# 5. Auctioneer: Runs the auction (with dynamic reserve threshold and coalition logic).
###############################################################################
class Auctioneer:
    def __init__(self, reserve_threshold: float = 25.0, decentralized: bool = False):
        self.reserve_threshold = reserve_threshold
        self.auction_log = []
        self.decentralized = decentralized
        if decentralized:
            self.blockchain = []  # list of blocks for blockchain–style logging

    def update_reserve_threshold(self, round_num, recent_auctions):
        """
        Adjust the reserve threshold based on recent auctions.
        For example, set threshold to 1.5× the average winning bid.
        """
        if recent_auctions:
            assigned_bids = [event['winning_bid'] for event in recent_auctions if event['status'] in ['assigned', 'assigned_coalition', 'assigned_above_threshold']]
            if assigned_bids:
                avg_bid = sum(assigned_bids) / len(assigned_bids)
                self.reserve_threshold = avg_bid * 1.5
        # Otherwise, leave unchanged.

    def add_block(self, event):
        """
        Append a new block (containing the event) to the blockchain.
        """
        prev_hash = self.blockchain[-1]['hash'] if self.blockchain else '0'
        block_data = json.dumps(event, sort_keys=True).encode()
        block_hash = hashlib.sha256(prev_hash.encode() + block_data).hexdigest()
        block = {'event': event, 'prev_hash': prev_hash, 'hash': block_hash}
        self.blockchain.append(block)

    def run_auction(self, task: Task, servers: list):
        """
        Run the auction for a given task. Always selects a winner.
        First, bids are collected from all servers. Then the server with the lowest bid is chosen.
        If that server is overloaded (load ratio > 0.8) and has a low–loaded partner in its coalition,
        a coalition assignment is attempted.
        Regardless of the reserve threshold, a winner is always assigned.
        If the winning bid exceeds the reserve threshold, the event status is marked 'assigned_above_threshold'.
        """
        bids = {}
        for server in servers:
            bid = server.compute_bid(task)
            bids[server.id] = bid
            logging.debug(f"Server {server.id} bid {bid:.2f} for task {task.id}")

        # Choose the server with the lowest bid (reverse auction)
        winner_id = min(bids, key=bids.get)
        winner = next(s for s in servers if s.id == winner_id)
        sorted_bids = sorted(bids.items(), key=lambda x: x[1])
        second_price = sorted_bids[1][1] if len(sorted_bids) > 1 else sorted_bids[0][1]

        # Check for coalition formation if the winner is overloaded.
                # Coalition formation: if winner is overloaded (above a lower threshold), try to offload half the task.
        if (winner.current_load / winner.capacity) > 0.6:
            partners = [
                s for s in servers 
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
                    'task_id': task.id,
                    'winner': (winner.id, partner.id),
                    'winning_bid': bids[winner.id],
                    'second_price': second_price,
                    'bids': bids,
                    'status': 'assigned_coalition'
                }
                self.auction_log.append(event)
                if self.decentralized:
                    self.add_block(event)
                logging.info(f"Auction for task {task.id}: Coalition assignment between Server {winner.id} and Server {partner.id}.")
                return (winner, partner), bids[winner.id], second_price

        # Normal assignment.
        # Instead of dropping the task when the winning bid exceeds the reserve threshold,
        # we always assign the task to a winner.
        if bids[winner.id] > self.reserve_threshold:
            status = 'assigned_above_threshold'
            logging.info(f"Auction for task {task.id}: Winning bid {bids[winner.id]:.2f} exceeds reserve threshold ({self.reserve_threshold:.2f}). Assigning task to Server {winner.id} with penalty.")
        else:
            status = 'assigned'
            logging.info(f"Auction for task {task.id}: Winner Server {winner.id} (bid {bids[winner.id]:.2f}).")
        event = {
            'task_id': task.id,
            'winner': winner.id,
            'winning_bid': bids[winner.id],
            'second_price': second_price,
            'bids': bids,
            'status': status
        }
        self.auction_log.append(event)
        if self.decentralized:
            self.add_block(event)
        winner.assign_task(task.load)
        return winner, bids[winner.id], second_price

###############################################################################
# 6. Simulation: Run network simulation over many rounds.
###############################################################################
def simulate_network(num_servers: int = 5,
                     num_rounds: int = 150,
                     base_tasks_per_round: float = 3.0,
                     processing_rate: float = 5.0,
                     random_seed: int = 42,
                     learning_mode: str = "continuous",  # Options: "continuous", "q-learning", "bandit"
                     agent_params: dict = None,
                     neural_bid_enabled: bool = False,
                     decentralized_auction: bool = False,
                     num_coalitions: int = 2):
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Initialize servers.
    servers = []
    for i in range(num_servers):
        capacity = random.uniform(50, 100)
        sensitivity = random.uniform(0.5, 2.0)
        risk_aversion = random.uniform(0, 0.5)
        coalition_id = random.randint(0, num_coalitions - 1)
        server = Server(
            id=i,
            capacity=capacity,
            sensitivity=sensitivity,
            target_load_ratio=0.5,
            learning_rate=0.1,
            learning_mode=learning_mode,
            agent_params=agent_params,
            neural_bid_enabled=neural_bid_enabled,
            risk_aversion=risk_aversion,
            coalition_id=coalition_id
        )
        servers.append(server)
        logging.info(f"Initialized Server {i}: capacity {capacity:.2f}, sensitivity {sensitivity:.2f}, "
                     f"risk_aversion {risk_aversion:.2f}, coalition_id {coalition_id}, learning_mode {learning_mode}")

    if decentralized_auction:
        auctioneer = Auctioneer(reserve_threshold=25.0, decentralized=True)
    else:
        auctioneer = Auctioneer(reserve_threshold=25.0, decentralized=False)

    # Prepare history records.
    server_load_history = {server.id: [] for server in servers}
    sensitivity_history = {server.id: [server.sensitivity] for server in servers}
    coalition_history = {server.id: [] for server in servers}
    reserve_threshold_history = []

    task_id_counter = 0

    # Metrics for fairness and social welfare.
    social_welfare_history = []
    fairness_history = []  # standard deviation of loads

    for round_num in range(num_rounds):
        logging.info(f"\n=== Round {round_num + 1} ===")
        # Non-stationary task arrival: modulate average tasks per round.
        amplitude = 2.0
        period = 50
        avg_tasks = base_tasks_per_round + amplitude * math.sin(2 * math.pi * round_num / period)
        avg_tasks = max(avg_tasks, 0)
        num_tasks = np.random.poisson(avg_tasks)
        tasks = []
        for _ in range(num_tasks):
            load = random.uniform(5, 20)
            task = Task(id=task_id_counter, load=load)
            tasks.append(task)
            task_id_counter += 1

        # Process each task.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks to the executor.
            futures = [executor.submit(process_task, task, auctioneer, servers) for task in tasks]
            # Wait for and process each result.
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                # (Optional) Process the result if needed.

        # End of round: process load and update strategies.
        for server in servers:
            server.process_load(processing_rate)
            server.update_strategy()
            server_load_history[server.id].append(server.current_load)
            sensitivity_history[server.id].append(server.sensitivity)
            coalition_history[server.id].append(server.coalition_id)

        # After updating server loads and strategies for this round:
        shapley_values = compute_shapley_values(servers)
        logging.info(f"Round {round_num + 1} - Shapley values: {shapley_values}")

        # Update reserve threshold based on auctions in this round.
        round_auctions = auctioneer.auction_log[-num_tasks:] if num_tasks > 0 else []
        auctioneer.update_reserve_threshold(round_num, round_auctions)
        reserve_threshold_history.append(auctioneer.reserve_threshold)

        # Compute fairness (std dev of loads) and social welfare.
        loads = [server.current_load for server in servers]
        fairness = np.std(loads)
        fairness_history.append(fairness)
        target_loads = [server.capacity * server.target_load_ratio for server in servers]
        social_welfare = -sum(abs(server.current_load - target) for server, target in zip(servers, target_loads))
        social_welfare_history.append(social_welfare)

    metrics = {
        'server_load_history': server_load_history,
        'sensitivity_history': sensitivity_history,
        'reserve_threshold_history': reserve_threshold_history,
        'fairness_history': fairness_history,
        'social_welfare_history': social_welfare_history,
        'auction_log': auctioneer.auction_log,
    }
    if decentralized_auction:
        metrics['blockchain'] = auctioneer.blockchain

    return metrics

def value_function(coalition):
    """
    A simple example: reward servers that have spare capacity.
    Here we define the value of a coalition as the sum of (capacity - current_load)
    for all servers in the coalition.
    """
    return sum(server.capacity - server.current_load for server in coalition)

def compute_shapley_values(servers, n_samples=1000):
    """
    Approximate the Shapley value for each server by sampling random orders.
    Returns a dictionary mapping server.id to its estimated Shapley value.
    """
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

    # Average over samples.
    for server_id in shapley:
        shapley[server_id] /= n_samples

    return shapley

###############################################################################
# 7. Plotting: Visualize simulation metrics.
###############################################################################
import os

def plot_metrics(metrics):
    # Turn off interactive mode so figures aren’t shown automatically.
    plt.ioff()
    
    # Get the maximum length from all histories
    max_length = max(
        len(metrics['server_load_history'].get(server_id, [])) 
        for server_id in metrics['server_load_history']
    )
    print(f"Max length of server load histories: {max_length}")
    
    # Correct way to get the first item from sensitivity_history
    first_sensitivity = next(iter(metrics['sensitivity_history'].values()), [])
    print(f"First sensitivity value length: {len(first_sensitivity)}")
    
    # Choose rounds length based on the minimum of these two lengths
    rounds = np.arange(min(max_length, len(first_sensitivity)))
    print(f"Rounds length: {len(rounds)}")
    
    # Truncate all histories to the length of rounds
    server_load_history = {server_id: loads[:len(rounds)] 
                           for server_id, loads in metrics['server_load_history'].items()}
    sensitivity_history = {server_id: sens[:len(rounds)] 
                           for server_id, sens in metrics['sensitivity_history'].items()}
    reserve_threshold_history = metrics['reserve_threshold_history'][:len(rounds)]
    fairness_history = metrics['fairness_history'][:len(rounds)]
    social_welfare_history = metrics['social_welfare_history'][:len(rounds)]

    # Print the lengths of the truncated histories for debugging.
    print(f"Truncated server load history lengths: {[len(loads) for loads in server_load_history.values()]}")
    print(f"Truncated sensitivity history lengths: {[len(sens) for sens in sensitivity_history.values()]}")
    print(f"Reserve threshold history length: {len(reserve_threshold_history)}")
    print(f"Fairness history length: {len(fairness_history)}")
    print(f"Social welfare history length: {len(social_welfare_history)}")
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot Server Loads.
    for server_id, loads in server_load_history.items():
        axs[0, 0].plot(rounds, loads, label=f"Server {server_id}")
    axs[0, 0].set_xlabel("Round")
    axs[0, 0].set_ylabel("Server Load")
    axs[0, 0].set_title("Evolution of Server Loads")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot Sensitivities.
    for server_id, sens in sensitivity_history.items():
        axs[0, 1].plot(rounds, sens, label=f"Server {server_id}")
    axs[0, 1].set_xlabel("Round")
    axs[0, 1].set_ylabel("Sensitivity")
    axs[0, 1].set_title("Evolution of Server Sensitivity")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot Reserve Threshold.
    axs[0, 2].plot(rounds, reserve_threshold_history, label="Reserve Threshold", color='purple')
    axs[0, 2].set_xlabel("Round")
    axs[0, 2].set_ylabel("Reserve Threshold")
    axs[0, 2].set_title("Evolution of Reserve Threshold")
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # Plot Fairness Metric.
    axs[1, 0].plot(rounds, fairness_history, label="Fairness (std dev)", color='green')
    axs[1, 0].set_xlabel("Round")
    axs[1, 0].set_ylabel("Fairness Metric")
    axs[1, 0].set_title("Fairness Over Rounds")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot Social Welfare.
    axs[1, 1].plot(rounds, social_welfare_history, label="Social Welfare", color='orange')
    axs[1, 1].set_xlabel("Round")
    axs[1, 1].set_ylabel("Social Welfare")
    axs[1, 1].set_title("Social Welfare Over Rounds")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Remove the extra subplot (position 2,2) since we only have 5 graphs.
    axs[1, 2].axis('off')

    # Adjust layout and save the figure.
    plt.tight_layout()
    save_path = './backend/static/simulation_summary.png'
    plt.savefig(save_path)
    print(f"Figure saved to '{save_path}' in directory: {os.getcwd()}")

    # Close the figure to free memory.
    plt.close(fig)



###############################################################################
# 8. Main: Run the simulation.
###############################################################################
def main():
    NUM_SERVERS = 5
    NUM_ROUNDS = 150
    BASE_TASKS_PER_ROUND = 3.0
    PROCESSING_RATE = 5.0

    # Choose learning mode: "continuous", "q-learning", or "bandit"
    LEARNING_MODE = "q-learning"
    agent_params = {"epsilon": 0.1, "alpha": 0.1, "gamma": 0.9}
    NEURAL_BID_ENABLED = True
    DECENTRALIZED_AUCTION = True
    NUM_COALITIONS = 2

    metrics = simulate_network(
        num_servers=NUM_SERVERS,
        num_rounds=NUM_ROUNDS,
        base_tasks_per_round=BASE_TASKS_PER_ROUND,
        processing_rate=PROCESSING_RATE,
        random_seed=42,
        learning_mode=LEARNING_MODE,
        agent_params=agent_params,
        neural_bid_enabled=NEURAL_BID_ENABLED,
        decentralized_auction=DECENTRALIZED_AUCTION,
        num_coalitions=NUM_COALITIONS
    )

    print("\nFinal Server Loads:")
    for server_id, loads in metrics['server_load_history'].items():
        print(f"  Server {server_id}: Final Load = {loads[-1]:.2f}")
    print("\nFinal Reserve Threshold:", metrics['reserve_threshold_history'][-1])
    print("\nFairness (std dev of loads) in final round:", metrics['fairness_history'][-1])
    print("\nSocial Welfare in final round:", metrics['social_welfare_history'][-1])
    print("\nFirst 5 Auction Events:")
    for event in metrics['auction_log'][:5]:
        print(event)
    if DECENTRALIZED_AUCTION:
        print("\nBlockchain (first 2 blocks):")
        for block in metrics['blockchain'][:2]:
            print(block)

    plot_metrics(metrics)

if __name__ == "__main__":
    main()
