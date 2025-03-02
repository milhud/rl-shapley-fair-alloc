from learning import *


class Server:
    def __init__(
        self,
        id: int,
        capacity: float,
        sensitivity: float,
        target_load_ratio: float = 0.5,
        learning_rate: float = 0.1,
        learning_mode: str = "continuous",  # Options: "continuous", "q-learning", "bandit"
        agent_params: dict = None,
        neural_bid_enabled: bool = True,
        risk_aversion: float = 0.1,
        coalition_id: int = 0,
    ):
        self.id = id
        self.capacity = capacity
        self.sensitivity = sensitivity
        self.current_load = 0.0
        self.target_load_ratio = target_load_ratio
        self.learning_rate = learning_rate

        self.risk_aversion = risk_aversion  # Factor that modulates risk sensitivity.
        self.coalition_id = coalition_id  # ID indicating coalition membership.

        self.bid_history = []
        self.task_history = []
        self.sensitivity_history = [sensitivity]

        # learning mode for adjusting sensitivity
        self.learning_mode = learning_mode
        self.learning_agent = LearningAgent(mode=learning_mode, **(agent_params or {}))
        self.last_state = None
        self.last_action = None
        self.last_error = None

        # neural bid predictor (dummy NN) for bid adjustment
        self.neural_bid_enabled = neural_bid_enabled
        if self.neural_bid_enabled:
            # for example, 4 input features and 3 actions
            self.dqn_agent = DQNAgent(input_dim=4, output_dim=3)
        else:
            self.dqn_agent = None

    def compute_bid(self, task) -> float:
        # calculate load_ratio and clamp it to a maximum of 1.0
        load_ratio = self.current_load / self.capacity
        load_ratio = min(load_ratio, 1.0)

        # compute a base bid using sensitivity and task load
        base_bid = self.sensitivity * load_ratio + task.load

        # only add risk adjustment if load_ratio exceeds target_load_ratio
        risk_adjustment = 1 + self.risk_aversion * max(
            0, (load_ratio - self.target_load_ratio)
        )
        base_bid *= risk_adjustment

        # add a small random noise
        noise = random.uniform(0, 0.1)

        # use neural bid adjustment if enabled
        if self.neural_bid_enabled and self.dqn_agent:
            # Create state tensor: [load_ratio, task.load, sensitivity, risk_aversion]
            state = torch.tensor(
                [[load_ratio, task.load, self.sensitivity, self.risk_aversion]],
                dtype=torch.float32,
            )
            action = self.dqn_agent.get_action(state)
            # -map action to a bid adjustment factor:
            if action == 0:
                adjustment = self.learning_rate  # increase bid adjustment
            elif action == 1:
                adjustment = -self.learning_rate  # decrease bid adjustment
            else:
                adjustment = 0.0  # no change
            bid = base_bid * (1 + adjustment) + noise
        else:
            bid = base_bid + noise

        # dampening the capacity multiplier: reduce from (1 + capacity/100) to (1 + capacity/200)
        bid *= 1 + (self.capacity / 200)

        # optionally, clamp the bid to a maximum value to avoid runaway bids.
        bid = min(bid, 1000.0)

        self.bid_history.append(bid)
        return bid

    def assign_task(self, task_load: float):
        self.current_load += task_load
        self.task_history.append(task_load)
        logging.info(
            f"Server {self.id} assigned task load {task_load:.2f}, new load: {self.current_load:.2f}"
        )

    def process_load(self, processing_rate: float):
        old_load = self.current_load
        self.current_load = max(0, self.current_load - processing_rate)
        if old_load != self.current_load:
            logging.debug(
                f"Server {self.id} processed load: {old_load:.2f} -> {self.current_load:.2f}"
            )

    def update_strategy(self):
        load_ratio = self.current_load / self.capacity
        current_state = int(load_ratio * 10)  # discretized state: 0 to 10
        current_error = abs(load_ratio - self.target_load_ratio)

        if self.learning_mode in ["q-learning", "bandit"]:
            if self.last_state is not None:
                reward = self.last_error - current_error  # positive if error decreased
                self.learning_agent.update(
                    self.last_state, self.last_action, reward, current_state
                )
                logging.debug(
                    f"Server {self.id} learning update: state {self.last_state}, action {self.last_action}, reward {reward:.3f}"
                )

            action = self.learning_agent.select_action(
                current_state
            )  # This is where action is selected

            old_sensitivity = self.sensitivity
            if action == "increase":
                self.sensitivity *= 1 + self.learning_rate
            elif action == "decrease":
                self.sensitivity *= 1 - self.learning_rate
            elif action == "none":
                pass
            self.sensitivity = max(self.sensitivity, 0.1)

            self.sensitivity_history.append(self.sensitivity)
            logging.debug(
                f"Server {self.id} applied action '{action}': sensitivity {old_sensitivity:.3f} -> {self.sensitivity:.3f}"
            )

            # store last state, action, and error for the next update
            self.last_state = current_state
            self.last_action = action
            self.last_error = current_error
        else:
            delta = load_ratio - self.target_load_ratio
            old_sensitivity = self.sensitivity
            self.sensitivity *= 1 + self.learning_rate * delta
            self.sensitivity = max(self.sensitivity, 0.1)
            self.sensitivity_history.append(self.sensitivity)
            logging.debug(
                f"Server {self.id} updated sensitivity continuously: {old_sensitivity:.3f} -> {self.sensitivity:.3f}"
            )
