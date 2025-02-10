from flask import Flask, request, jsonify, send_from_directory
import json
import logging
from simulation import *  # Import the simulation function
import os
import subprocess  
from dotenv import load_dotenv

app = Flask(__name__)

# Initialize logger
logging.basicConfig(level=logging.INFO)

# This will store tasks temporarily
tasks = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This gets the directory where api.py is located
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "backend"))
SERVERS_JSON_PATH = os.path.join(PROJECT_DIR, "backend", "servers.json")




@app.route('/app/demonstration', methods=['GET'])
def run_simulation():
    
    SIMULATION_SCRIPT = os.path.join(PROJECT_DIR, "backend", "poc_simulation.py")  # Full path to script
    IMAGE_PATH = os.path.join(PROJECT_DIR, "backend", "simulation_summary.png")
    STATIC_PATH = os.path.join(PROJECT_DIR, "backend", "static", "simulation_summary.png")
    
    load_dotenv()
    PYTHON_FILE_PATH = os.getenv("ABSOLUTE_FILE_PATH")

    try:
        # Run opc_simulation.py and wait for it to complete
        subprocess.run([PYTHON_FILE_PATH, SIMULATION_SCRIPT], check=True)

        # Define the path to the simulation summary image
        image_path = os.path.join(STATIC_PATH)

        # Check if the image was generated
        if not os.path.exists(image_path):
            return jsonify({"error": "Simulation completed, but image not found."}), 500

        # Serve the generated image
        return send_from_directory("static", "simulation_summary.png")

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Simulation failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# Load server list from servers.json or create it if missing
def load_servers():
    if not os.path.exists(SERVERS_JSON_PATH):
        logging.warning("servers.json not found. Creating a new one with an empty list.")
        save_servers([])  # Create the file with an empty list
    
    try:
        with open(SERVERS_JSON_PATH, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        logging.warning("servers.json is empty or contains invalid JSON. Resetting to an empty list.")
        save_servers([])  # Reset to an empty list
        return []

# Write updated server data to servers.json
def save_servers(servers):
    try:
        with open(SERVERS_JSON_PATH, "w") as file:
            json.dump(servers, file, indent=4)
    except Exception as e:
        logging.error(f"Failed to save servers.json: {e}")


# Endpoint to accept a new task and return the assigned server's IP address
@app.route("/task", methods=["POST"])
def submit_task():
    data = request.json  # Get the task details from the request

    if not data or "load" not in data:
        return jsonify({"error": "Invalid task format, 'load' is required"}), 400

    # Assign the task using auction logic
    ip = handle_task_submission(data)  # Ensure this function interfaces with app.py

    return jsonify({"assigned_server": ip})

# Endpoint to update servers.json with new server data
@app.route("/update_servers", methods=["POST"])
def update_servers():
    new_servers = request.json  # Expecting a JSON list of servers

    if not isinstance(new_servers, list):
        return jsonify({"error": "Invalid format. Expected a list of servers."}), 400

    # Save the new server list
    save_servers(new_servers)

    return jsonify({"message": "Server list updated successfully."}), 200

@app.route('/simulate', methods=['POST'])
def simulate():
    """
    Start a simulation of network load balancing with user-defined parameters.
    """
    params = request.json

    # Set default values if the parameter is missing from the request
    num_servers = params.get('num_servers', 5)
    num_rounds = params.get('num_rounds', 150)
    base_tasks_per_round = params.get('base_tasks_per_round', 3.0)
    processing_rate = params.get('processing_rate', 5.0)
    random_seed = params.get('random_seed', 42)
    learning_mode = params.get('learning_mode', 'continuous')
    agent_params = params.get('agent_params', {})
    neural_bid_enabled = params.get('neural_bid_enabled', False)
    decentralized_auction = params.get('decentralized_auction', False)
    num_coalitions = params.get('num_coalitions', 2)

    # Simulate the network with the provided parameters
    metrics = simulate_network(
        num_servers=num_servers,
        num_rounds=num_rounds,
        base_tasks_per_round=base_tasks_per_round,
        processing_rate=processing_rate,
        random_seed=random_seed,
        learning_mode=learning_mode,
        agent_params=agent_params,
        neural_bid_enabled=neural_bid_enabled,
        decentralized_auction=decentralized_auction,
        num_coalitions=num_coalitions
    )

    # Select the server with the least current load
    server_with_least_load = min(metrics['server_load_history'], key=lambda server_id: min(metrics['server_load_history'][server_id]))

    return jsonify({
        'server_id': server_with_least_load,
        'metrics': metrics
    })


if __name__ == '__main__':
    app.run(debug=True)
