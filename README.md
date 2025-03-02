This is a proof of concept game-theoretic load balancer. To launch the frontend, cd into the frontend directory and run:

```
npm start
```

and for the backend, cd into the backend directory and run:

```
python api.py
```

From the frontend, you can send tasks with different complexities, and the program will return the server in which the task is assigned. Additionally, you can choose to run a simulation with a predefined numbre of tasks being sent to a predefined number of servers.

Here is a brief explanation of the core concepts behind this program:


### 1. Initialization

- **Server Setup:**  
  Each server is initialized with:
  - A **capacity** (maximum workload it can handle)
  - A **sensitivity** value (affects how aggressively it bids)
  - A **risk aversion** parameter (influences bid adjustments when nearing overload)

- **Auctioneer Module:**  
  An auctioneer is created to manage the bidding process, including maintaining a **reserve threshold** which is updated after each round.

### 2. Task Generation

- **Dynamic Task Arrival:**  
  Tasks are generated in rounds using a sinusoidal function. This simulates non-stationary workload conditions where the number of tasks varies over time.
  
- **Task Characteristics:**  
  Each task includes parameters such as:
  - **Load:** The amount of work required.
  - **SLA Level:** Priority levels (e.g., gold, silver, bronze) that may influence bidding.

### 3. Bid Computation

Each server computes a bid for incoming tasks through several stages:

- **Calculating the Load Ratio:**  
  This represents how busy a server is relative to its capacity.  
  ```math
  \text{load ratio} = \min\left(\frac{\text{current\_load}}{\text{capacity}}, 1.0\right)
  ```
  
- **Computing the Base Bid:**  
  The base bid is determined by the server's sensitivity, its current load, and the task's load.  
  ```math
  \text{base bid} = \text{sensitivity} \times \text{load ratio} + \text{task.load}
  ```
  
- **Risk Adjustment:**  
  To discourage overloaded servers from taking on more work, a risk factor is applied:  
  ```math
  \text{risk factor} = 1 + \text{risk\_aversion} \times \max\left(0, \text{load ratio} - \text{target load ratio}\right)
  ```  
  This yields an adjusted bid:
  ```math
  \text{adjusted bid} = \text{base bid} \times \text{risk factor}
  ```
  
- **Dampening and Clamping:**  
  Finally, the bid is scaled based on capacity and clamped to a maximum value to prevent runaway bids:
  ```math
  \text{final bid} = \min\left( \text{bid} \times \left(1 + \frac{\text{capacity}}{200}\right), 1000 \right)
  ```

### 4. Auction Process

- **Reverse Auction Mechanism:**  
  The auctioneer collects bids from all servers and selects the one with the lowest bid, using a second-price mechanism to promote truthful bidding.
  
- **Coalition Formation:**  
  If a server is overloaded (i.e., its load ratio is too high), the system may form a coalition with another server. Fairness in splitting the workload is ensured with the calculated Shapley values.

### 5. Learning and Updates

Servers continuously refine their bidding strategies based on past outcomes through two main approaches:

- **Q-Learning:**  
  Each server maintains a Q-table that maps state-action pairs. The Q-learning update rule is:
  ```math
  Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
  ```
  - $\( \alpha \)$ is the learning rate.
  - $\( r \)$ is the reward (e.g., reduction in load mismatch).
  - $\( \gamma \)$ is the discount factor.
  
- **Multi-Armed Bandit:**  
  For simpler, state-independent updates, a bandit approach is used:
  ```math
  \text{value}_a \leftarrow \text{value}_a + \frac{1}{n_a} \left( r - \text{value}_a \right)
  ```
  - \( n_a \) is the count of how many times action \( a \) has been selected.

### 6. Reserve Threshold Adjustment

After each auction round, the auctioneer updates the reserve threshold to reflect the current bidding environment:
```math
\text{reserve threshold} = 1.5 \times \text{average winning bid}
```

### 7. Post-Auction Processing

- Task Processing: 
  Once tasks are assigned, servers process them, gradually reducing their load over time.
  
- Metrics Collection: 
  The simulation collects data on server loads, bidding sensitivity changes, fairness (via standard deviation of loads), and overall social welfare. This helps in evaluating system performance and fairness in task distribution.


You can run poc_simulation.py to create a simulation, simulating sending an arbitrary amount of tasks to an arbitrary amount of servers.
