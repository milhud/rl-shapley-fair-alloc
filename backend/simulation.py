import random
import numpy as np
import logging
import math
import concurrent.futures
import matplotlib.pyplot as plt
import os
from auction import *
from server import *
from task import *

auctioneer = None
servers = []

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # This gets the directory where api.py is located
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "backend"))
SERVERS_JSON_PATH = os.path.join(PROJECT_DIR, "backend", "servers.json")


def simulate_network(
    num_servers: int = 5,
    num_rounds: int = 150,
    base_tasks_per_round: float = 3.0,
    processing_rate: float = 5.0,
    random_seed: int = 42,
    learning_mode: str = "continuous",  # Options: "continuous", "q-learning", "bandit"
    agent_params: dict = None,
    neural_bid_enabled: bool = False,
    decentralized_auction: bool = False,
    num_coalitions: int = 2,
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    global servers  # allow modification of global variables

    from auction import Auctioneer
    from server import Server

    # initialize servers
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
            coalition_id=coalition_id,
        )
        servers.append(server)
        logging.info(
            f"Initialized Server {i}: capacity {capacity:.2f}, sensitivity {sensitivity:.2f}, "
            f"risk_aversion {risk_aversion:.2f}, coalition_id {coalition_id}, learning_mode {learning_mode}"
        )

    if decentralized_auction:
        auctioneer = Auctioneer(reserve_threshold=25.0, decentralized=True)
    else:
        auctioneer = Auctioneer(reserve_threshold=25.0, decentralized=False)

    # prepare
    server_load_history = {server.id: [] for server in servers}
    sensitivity_history = {server.id: [server.sensitivity] for server in servers}
    coalition_history = {server.id: [] for server in servers}
    reserve_threshold_history = []
    auction_event_count_history = []  # To track the auction event counts

    task_id_counter = 0

    # metrics for fairness and social welfare
    social_welfare_history = []
    fairness_history = []  # standard deviation of loads

    for round_num in range(num_rounds):
        logging.info(f"\n=== Round {round_num + 1} ===")
        # non-stationary task arrival: modulate average tasks per round
        amplitude = 2.0
        period = 50
        avg_tasks = base_tasks_per_round + amplitude * math.sin(
            2 * math.pi * round_num / period
        )
        avg_tasks = max(avg_tasks, 0)
        num_tasks = np.random.poisson(avg_tasks)
        tasks = []
        for _ in range(num_tasks):
            load = random.uniform(5, 20)
            task = Task(id=task_id_counter, load=load)
            tasks.append(task)
            task_id_counter += 1

        # process each task
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks to the executor
            futures = [
                executor.submit(process_task, task, auctioneer, servers)
                for task in tasks
            ]
            # wait for and process each result
            for future in concurrent.futures.as_completed(futures):
                result = future.result()

        # end of round: process load and update strategies
        for server in servers:
            server.process_load(processing_rate)
            server.update_strategy()
            server_load_history[server.id].append(server.current_load)
            sensitivity_history[server.id].append(server.sensitivity)
            coalition_history[server.id].append(server.coalition_id)

        # after updating server loads and strategies for this round - 
        shapley_values = compute_shapley_values(servers)
        logging.info(f"Round {round_num + 1} - Shapley values: {shapley_values}")

        # update reserve threshold based on auctions in this round - 
        round_auctions = auctioneer.auction_log[-num_tasks:] if num_tasks > 0 else []
        auctioneer.update_reserve_threshold(round_num, round_auctions)
        reserve_threshold_history.append(auctioneer.reserve_threshold)

        # track the number of auction events in this round
        auction_event_count = len(auctioneer.auction_log) if num_tasks > 0 else 0
        auction_event_count_history.append(auction_event_count)

        # compute fairness (std dev of loads) and social welfare
        loads = [server.current_load for server in servers]
        fairness = np.std(loads)
        fairness_history.append(fairness)
        target_loads = [
            server.capacity * server.target_load_ratio for server in servers
        ]
        social_welfare = -sum(
            abs(server.current_load - target)
            for server, target in zip(servers, target_loads)
        )
        social_welfare_history.append(social_welfare)

    # build the final metrics dictionary
    metrics = {
        "server_load_history": server_load_history,
        "sensitivity_history": sensitivity_history,
        "reserve_threshold_history": reserve_threshold_history,
        "fairness_history": fairness_history,
        "social_welfare_history": social_welfare_history,
        "auction_log": auctioneer.auction_log,
        "auction_event_count_history": auction_event_count_history,  # Add auction event counts
    }
    if decentralized_auction:
        metrics["blockchain"] = auctioneer.blockchain

    # return metrices dictionary
    logging.debug("Final metrics: %s", metrics)
    return metrics

""" again because I forgot which one worked """
import random
import numpy as np
import logging
import math
import concurrent.futures
import matplotlib.pyplot as plt
from auction import Auctioneer
from server import Server
from task import Task
from auction import compute_shapley_values


def simulate_network(
    num_servers: int = 5,
    num_rounds: int = 150,
    base_tasks_per_round: float = 3.0,
    processing_rate: float = 5.0,
    random_seed: int = 42,
    learning_mode: str = "continuous",  # Options: "continuous", "q-learning", "bandit"
    agent_params: dict = None,
    neural_bid_enabled: bool = False,
    decentralized_auction: bool = False,
    num_tasks_per_round=5,
    num_coalitions: int = 2,
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    # initialize servers
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
            coalition_id=coalition_id,
        )
        servers.append(server)
        logging.info(
            f"Initialized Server {i}: capacity {capacity:.2f}, sensitivity {sensitivity:.2f}, "
            f"risk_aversion {risk_aversion:.2f}, coalition_id {coalition_id}, learning_mode {learning_mode}"
        )

    if decentralized_auction:
        auctioneer = Auctioneer(reserve_threshold=25.0, decentralized=True)
    else:
        auctioneer = Auctioneer(reserve_threshold=25.0, decentralized=False)

    # prepare records
    server_load_history = {server.id: [] for server in servers}
    sensitivity_history = {server.id: [server.sensitivity] for server in servers}
    coalition_history = {server.id: [] for server in servers}
    reserve_threshold_history = []
    auction_event_count_history = []  # To track the auction event counts

    task_id_counter = 0

    for round_num in range(num_rounds):
        logging.info(f"\n=== Round {round_num + 1} ===")

        # create tasks for the current round
        tasks = []
        for _ in range(num_tasks_per_round):
            load = random.uniform(5, 20)  # random load for task
            task = Task(id=task_id_counter, load=load)
            tasks.append(task)
            task_id_counter += 1

            # auction and task assignment
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(auctioneer.run_auction, task, servers)
                    for task in tasks
                ]
                # wait for and process each result
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    winner, winning_bid, second_price = result

                    # hndle the task assignment outcome
                    # logging.info(f"Task assigned to Server {winner.id} (Winning bid: {winning_bid:.2f}, Second price: {second_price:.2f})")

            # update the servers' load after tasks have been assigned
            for server in servers:
                server.process_load(processing_rate)

            # optionally, handle coalitions and task offloading
            # this part of the logic will be managed within the auction method itself
            logging.info(f"Round {round_num + 1} completed.")

            # update server strategy, process loads, etc.
            for server in servers:
                server.process_load(processing_rate)
                server.update_strategy()
                server_load_history[server.id].append(server.current_load)
                sensitivity_history[server.id].append(server.sensitivity)
                coalition_history[server.id].append(server.coalition_id)

    # metrics for fairness and social welfare.
    social_welfare_history = []
    fairness_history = []  # standard deviation of loads

    for round_num in range(num_rounds):
        logging.info(f"\n=== Round {round_num + 1} ===")
        # non-stationary task arrival: modulate average tasks per round.
        amplitude = 2.0
        period = 50
        avg_tasks = base_tasks_per_round + amplitude * math.sin(
            2 * math.pi * round_num / period
        )
        avg_tasks = max(avg_tasks, 0)
        num_tasks = np.random.poisson(avg_tasks)
        tasks = []
        for _ in range(num_tasks):
            load = random.uniform(5, 20)
            task = Task(id=task_id_counter, load=load)
            tasks.append(task)
            task_id_counter += 1

        # process each task
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # submit all tasks to the executor.
            futures = [
                executor.submit(process_task, task, auctioneer, servers)
                for task in tasks
            ]
            # wait for and process each result.
            for future in concurrent.futures.as_completed(futures):
                result = future.result()

        # end of round: process load and update strategies
        for server in servers:
            server.process_load(processing_rate)
            server.update_strategy()
            server_load_history[server.id].append(server.current_load)
            sensitivity_history[server.id].append(server.sensitivity)
            coalition_history[server.id].append(server.coalition_id)

        # after updating server loads and strategies for this round -
        shapley_values = compute_shapley_values(servers)
        logging.info(f"Round {round_num + 1} - Shapley values: {shapley_values}")

        # update reserve threshold based on auctions in this round
        round_auctions = auctioneer.auction_log[-num_tasks:] if num_tasks > 0 else []
        auctioneer.update_reserve_threshold(round_num, round_auctions)
        reserve_threshold_history.append(auctioneer.reserve_threshold)

        # track the number of auction events in this round
        auction_event_count = len(auctioneer.auction_log) if num_tasks > 0 else 0
        auction_event_count_history.append(auction_event_count)

        # compute fairness (std dev of loads) and social welfare
        loads = [server.current_load for server in servers]
        fairness = np.std(loads)
        fairness_history.append(fairness)
        target_loads = [
            server.capacity * server.target_load_ratio for server in servers
        ]
        social_welfare = -sum(
            abs(server.current_load - target)
            for server, target in zip(servers, target_loads)
        )
        social_welfare_history.append(social_welfare)

    # build the final metrics dictionary
    metrics = {
        "server_load_history": server_load_history,
        "sensitivity_history": sensitivity_history,
        "reserve_threshold_history": reserve_threshold_history,
        "fairness_history": fairness_history,
        "social_welfare_history": social_welfare_history,
        "auction_log": auctioneer.auction_log,
        "auction_event_count_history": auction_event_count_history,  # Add auction event counts
    }
    plot_metrics(metrics, save_path="simulation_metrics.png")
    if decentralized_auction:
        metrics["blockchain"] = auctioneer.blockchain

    # IMPORTANT: Return the metrics dictionary so the function doesn't return None
    logging.debug("Final metrics: %s", metrics)
    return metrics


def auction_task(task, auctioneer, servers):
    # each server places a bid based on its load and available capacity
    bids = []
    for server in servers:
        # bidding strategy could be based on load, capacity, and other factors
        bid = server.current_load / server.capacity  # Simple example
        bids.append((server, bid))

    # auctioneer selects the server with the lowest bid
    winning_server = min(bids, key=lambda x: x[1])[0]

    # assign the task to the winning server
    logging.info(f"Task {task.id} assigned to Server {winning_server.id}")
    winning_server.current_load += (
        task.load
    )  # update the server's load after task assignment
    return winning_server.id  # return the winning server's ID


def plot_metrics(metrics, save_path="simulation_metrics.png"):
    # use the length of fairness_history or any other metric that contains the number of rounds
    rounds = np.arange(len(metrics["fairness_history"]))

    plt.figure(figsize=(15, 10))

    # plot fairness, reserve threshold, social welfare, and auction event counts on the same graph
    plt.subplot(2, 2, 1)
    plt.plot(
        rounds, metrics["fairness_history"], label="Fairness (std dev)", color="green"
    )
    plt.plot(
        rounds,
        metrics["reserve_threshold_history"],
        label="Reserve Threshold",
        color="purple",
    )
    plt.plot(
        rounds,
        metrics["social_welfare_history"],
        label="Social Welfare",
        color="orange",
    )
    plt.plot(
        rounds,
        metrics["auction_event_count_history"],
        label="Auction Event Count",
        color="blue",
        linestyle="--",
    )

    plt.xlabel("Round")
    plt.ylabel("Value")
    plt.title("Fairness, Reserve Threshold, Social Welfare, and Auction Events")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # save the figure to the specified path
    plt.savefig(save_path)

    logging.info(f"Graph saved as {save_path}")
    plt.close()  # close the figure to release memory


def update_server_capacity(server_id, complexity, filename=SERVERS_JSON_PATH):
    """Update the server's capacity based on task complexity and persist changes safely."""

    # load existing server data
    try:
        with open(SERVERS_JSON_PATH, "r") as f:
            servers = json.load(f)  # read the JSON file correctly
    except (FileNotFoundError, json.JSONDecodeError):
        print("Error: servers.json not found or is corrupted.")
        return

    # check if servers data is a list
    if not isinstance(servers, list):
        print("Error: servers.json does not contain a valid list of servers.")
        return

    # modify the correct server
    updated = False
    for server in servers:
        if server.get("id") == server_id:  # Use .get() to avoid KeyError
            if "capacity" in server:  # Ensure capacity exists before modifying
                server["capacity"] -= complexity * 0.1  # Adjust as needed
                updated = True
                print(
                    f"✅ Updated Server {server_id}: New Capacity = {server['capacity']}"
                )
                break  # Stop once we find the server

    if not updated:
        print(f"Warning: No server found with ID {server_id}.")
        return

    # safely write the updated servers list back to servers.json
    try:
        with open(SERVERS_JSON_PATH, "w") as f:
            json.dump(servers, f, indent=4)  #prroperly format the JSON
    except IOError:
        print("Error: Could not write to servers.json.")


def handle_task_submission(task_data):
    # ensure task_data has required fields
    if "id" not in task_data or "load" not in task_data:
        return {"error": "Invalid task data. 'id' and 'load' are required."}

    task = Task(id=task_data["id"], load=task_data["load"])  # Create Task object

    # correct initialization of auctioneer
    auctioneer = Auctioneer(reserve_threshold=25.0, decentralized=False)

    # correctly pass auctioneer as a parameter, ensuring auctioneer is an object
    server_id = assign_task(task, servers, auctioneer)

    if server_id is not None:
        return {
            "assigned_server": server_id,
            "message": f"Task {task.id} assigned to Server {server_id}",
        }
    else:
        return {"error": "No server could be assigned"}


def assign_task(task, servers, auctioneer):
    result = auctioneer.run_auction(
        task, servers
    )  # ensure run_auction returns a winner

    if result:
        winning_server, winning_bid, second_price = result
        winning_server.current_load += task.load  # Assign task to server
        logging.info(
            f"Task {task.id} assigned to Server {winning_server.id} (Winning bid: {winning_bid:.2f})"
        )
        update_server_capacity(winning_server.id, task.complexity)
        return winning_server.id
    else:
        logging.info(f"Task {task.id} could not be assigned.")
        return None
