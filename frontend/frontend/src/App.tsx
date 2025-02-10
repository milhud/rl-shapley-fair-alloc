import React, { useState, useEffect } from "react";
import "./App.css";

interface Server {
  id: number;
  capacity: number;
  sensitivity: number;
}

const App: React.FC = () => {
  // State for number of servers
  const [numServers, setNumServers] = useState<number>(1);
  // State for the servers array (each server holds id, capacity, sensitivity)
  const [servers, setServers] = useState<Server[]>([]);
  // Console output text
  const [output, setOutput] = useState<string>('');
  // State for task submission data
  const [task, setTask] = useState<{ id: number; load: number; complexity: string; size: string }>({
    id: 0,
    load: 0,
    complexity: "",
    size: "",
  });
  // State for current server metrics (updated only after a successful update)
  const [currentMetrics, setCurrentMetrics] = useState<Server[]>([]);
  // State for simulation image URL
  const [simulationImage, setSimulationImage] = useState<string | null>(null);

  // When the number of servers changes, create a new server array.
  useEffect(() => {
    // Create an array of length numServers with default values.
    const newServers = Array.from({ length: numServers }, () => ({
      id: 0,
      capacity: 0,
      sensitivity: 0,
    }));
    setServers(newServers);
  }, [numServers]);

  // Handles changes to any server input field.
  const handleServerChange = (index: number, field: keyof Server, value: string) => {
    // If the field is empty, leave it as empty; otherwise, parse to number.
    const parsedValue =
      value === ""
        ? ""
        : field === "id"
        ? parseInt(value, 10)
        : parseFloat(value);
    const updatedServers = [...servers];
    updatedServers[index] = { ...updatedServers[index], [field]: parsedValue };
    setServers(updatedServers);
  };

  // Handles changes to the number of servers slider.
  const handleServerCountChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setNumServers(parseInt(event.target.value, 10));
  };

  // Handles changes in the task input fields.
  const handleTaskChange = (field: keyof typeof task, value: string) => {
    setTask((prevTask) => ({
      ...prevTask,
      [field]:
        value === ""
          ? ""
          : field === "id"
          ? parseInt(value, 10)
          : field === "load"
          ? parseFloat(value)
          : value,
    }));
  };

  // Called when the Update Servers button is clicked.
  const handleUpdateServers = async () => {
    setOutput("Updating servers...");
    try {
      const response = await fetch("http://localhost:5000/update_servers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(servers),
      });
      if (response.ok) {
        // We assume a successful update: update the current metrics to show the new server data.
        setCurrentMetrics(servers);
        setOutput("Servers successfully updated.");
      } else {
        setOutput("Error updating servers.");
      }
    } catch {
      setOutput("Network error while updating servers.");
    }
  };

  // Called when the Send Task button is clicked.
  const handleSendTask = async () => {
    setOutput("Sending task...");
    try {
      const response = await fetch("http://localhost:5000/task", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(task),
      });
      if (response.ok) {
        const taskResponse = await response.json();
        setOutput(`Task Submitted: ${JSON.stringify(taskResponse, null, 2)}`);
      } else {
        setOutput("Error submitting task.");
      }
    } catch {
      setOutput("Network error while submitting task.");
    }
  };

  // Called when the Run Simulation button is clicked.
  const handleRunSimulation = async () => {
    setOutput("Running simulation...");
    setSimulationImage(null); // Clear any previous image.
    try {
      const response = await fetch("http://localhost:5000/app/demonstration");
      if (response.ok) {
        setSimulationImage("http://localhost:5000/static/simulation_summary.png");
        setOutput("Simulation complete.");
      } else {
        setOutput("Error running simulation.");
      }
    } catch {
      setOutput("Network error while running simulation.");
    }
  };

  return (
    <div className="app">
      {/* Left Column: Server Input and Update Servers */}
      <div className="left-column">
        <div className="server-controls">
          <label>Number of Servers</label>
          <input
            type="range"
            min="1"
            max="10"
            value={numServers}
            onChange={handleServerCountChange}
          />
          <span>{numServers}</span>
        </div>
        <div className="server-list">
          {servers.map((server, index) => (
            <div key={index} className="server-item">
              <h3>Server {index + 1}</h3>
              <input
                type="number"
                placeholder="Server ID (Integer)"
                value={server.id === 0 ? "" : server.id}
                onChange={(e) => handleServerChange(index, "id", e.target.value)}
              />
              <input
                type="number"
                step="0.1"
                placeholder="Capacity (Float)"
                value={server.capacity === 0 ? "" : server.capacity}
                onChange={(e) => handleServerChange(index, "capacity", e.target.value)}
              />
              <input
                type="number"
                step="0.1"
                placeholder="Sensitivity (Float)"
                value={server.sensitivity === 0 ? "" : server.sensitivity}
                onChange={(e) => handleServerChange(index, "sensitivity", e.target.value)}
              />
            </div>
          ))}
        </div>
        <button className="blue-button" onClick={handleUpdateServers}>
          Update Servers
        </button>
      </div>

      {/* Middle Column: Simulation and Displaying Current Metrics */}
      <div className="middle-column">
        <button className="blue-button" onClick={handleRunSimulation}>
          Run Demonstration
        </button>
        {simulationImage && (
          <div className="simulation-image">
            <img
              src={simulationImage}
              alt="Simulation Summary"
              onClick={() => window.open(simulationImage, "_blank")}
            />
            <p className="expand-note">Click to expand image</p>
          </div>
        )}
        <h2>Current Server Metrics</h2>
        <table className="server-metrics-table">
          <thead>
            <tr>
              <th>Server</th>
              <th>ID</th>
              <th>Capacity</th>
              <th>Sensitivity</th>
            </tr>
          </thead>
          <tbody>
            {currentMetrics.map((server, index) => (
              <tr key={index}>
                <td>Server {index + 1}</td>
                <td>{server.id}</td>
                <td>{server.capacity}</td>
                <td>{server.sensitivity}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Right Column: Task Submission */}
      <div className="right-column">
        <h2>Submit Task</h2>
        <div className="task-inputs">
          <input
            type="number"
            placeholder="Task ID (Integer)"
            value={task.id === 0 ? "" : task.id}
            onChange={(e) => handleTaskChange("id", e.target.value)}
          />
          <input
            type="number"
            step="0.1"
            placeholder="Load (Float)"
            value={task.load === 0 ? "" : task.load}
            onChange={(e) => handleTaskChange("load", e.target.value)}
          />
          <input
            type="text"
            placeholder="Complexity"
            value={task.complexity}
            onChange={(e) => handleTaskChange("complexity", e.target.value)}
          />
          <input
            type="text"
            placeholder="Size"
            value={task.size}
            onChange={(e) => handleTaskChange("size", e.target.value)}
          />
          <button className="blue-button task-submit-button task-inputs"  style={{ marginTop: "10px" }} onClick={handleSendTask}>
            Send Task
          </button>
        </div>
        <div className="output-box">
        <h3>Console Output</h3>
        <pre className="console-output">{output}</pre>
      </div>

      <a 
        href="https://github.com/milhud/game-theoretic-load-balancer" 
        target="_blank" 
        rel="noopener noreferrer"
        className="blue-button"
        style={{ marginTop: "15px" }}
      >
        HELP / INFO
      </a>

      </div>
    </div>
  );
};

export default App;
