import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import logging
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