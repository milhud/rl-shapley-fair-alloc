# test_simulation.py
import unittest
import logging
from simulation import simulate_network  # Assuming simulate_network is in simulation.py

class TestSimulation(unittest.TestCase):

    def setUp(self):
        self.num_servers = 5
        self.num_rounds = 150
        self.base_tasks_per_round = 3.0
        self.processing_rate = 5.0
        self.learning_mode = "q-learning"
        self.agent_params = {"epsilon": 0.1, "alpha": 0.1, "gamma": 0.9}
        self.neural_bid_enabled = True
        self.decentralized_auction = True
        self.num_coalitions = 2
        self.random_seed = 42

    def test_simulate_network_metrics(self):
        # Set logging level to DEBUG to see debug logs
        logging.basicConfig(level=logging.DEBUG)
        
        # Run the simulation and capture metrics
        metrics = simulate_network(
            num_servers=self.num_servers,
            num_rounds=self.num_rounds,
            base_tasks_per_round=self.base_tasks_per_round,
            processing_rate=self.processing_rate,
            random_seed=self.random_seed,
            learning_mode=self.learning_mode,
            agent_params=self.agent_params,
            neural_bid_enabled=self.neural_bid_enabled,
            decentralized_auction=self.decentralized_auction,
            num_coalitions=self.num_coalitions
        )
        
        # Check and log the result
        logging.debug("Metrics returned from simulate_network: %s", metrics)
        
        # Test the returned metrics
        self.assertIsNotNone(metrics, "The metrics should not be None.")
        self.assertIn('server_load_history', metrics, "server_load_history not found in metrics")
        self.assertIn('reserve_threshold_history', metrics, "reserve_threshold_history not found in metrics")
        self.assertIn('fairness_history', metrics, "fairness_history not found in metrics")
        self.assertIn('social_welfare_history', metrics, "social_welfare_history not found in metrics")
        self.assertIn('auction_log', metrics, "auction_log not found in metrics")
        if self.decentralized_auction:
            self.assertIn('blockchain', metrics, "blockchain not found in metrics")

        # Log the final values
        logging.debug("Final server loads: %s", metrics['server_load_history'])
        logging.debug("Final reserve threshold: %s", metrics['reserve_threshold_history'])
        logging.debug("Final social welfare: %s", metrics['social_welfare_history'])

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()
