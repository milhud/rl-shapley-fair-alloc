# Reinforcement, Shapley, Fair Allocation Load Balancing - Game-Theoretic Load Balancer

This project implements a load balancer that uses auction-based task allocation combined with reinforcement learning and game-theoretic principles. Tasks are assigned to servers based on dynamically computed bids, with the system continuously adapting its parameters to balance loads efficiently.

## Running the Program

### Frontend
Start the frontend with:
```sh
cd frontend
npm start
```

### Backend
Start the backend with:
```sh
cd backend
python api.py
```

You can also run a simulation using:
```sh
python poc_simulation.py
```
This simulates sending a set number of tasks to a group of servers.

## How the Program Works

### 1. Server Initialization
- **Servers** are created with:
  - **Capacity** ($C$): Maximum workload.
  - **Sensitivity** ($s$): Determines bid aggressiveness.
  - **Risk Aversion** ($r$): Adjusts bids when near overload.
- An **auctioneer** is set up to manage task assignments and update the reserve threshold.

### 2. Task Generation
- **Tasks** arrive in rounds.
- Each task has:
  - **Load Demand** ($L_t$)
  - **SLA Level** ($\lambda$)
- Task arrival can be modeled by a function (e.g., sinusoidal):

$$
N(t) = N_{\max} \cdot \sin(t)
$$

### 3. Bid Computation by Each Server
Each server computes its bid in several steps:

#### a. Calculate Load Ratio
The load ratio indicates how busy a server is:
  
$$
R = \min\left(\frac{L_{\text{current}}}{C},\, 1.0\right)
$$

#### b. Compute Base Bid
The base bid combines the server’s sensitivity, load ratio, and the task’s load:
  
$$
B_{\text{base}} = s \cdot R + L_t
$$

#### c. Apply Risk Adjustment
To discourage overloaded servers from taking on extra tasks:
  
$$
F = 1 + r \cdot \max\left(0,\, R - R_{\text{target}}\right)
$$

Then, the adjusted bid is:

$$
B_{\text{adj}} = B_{\text{base}} \cdot F
$$

#### d. Final Bid Scaling & Clamping
Finally, the bid is scaled by capacity and capped:
  
$$
B_{\text{final}} = \min\left( B_{\text{adj}} \cdot \left(1 + \frac{C}{200}\right),\, 1000 \right)
$$

### 4. Auction Process
- **Reverse Auction:** All servers submit bids, and the server with the lowest bid wins.
- **Second-Price Rule:** The winning bid is influenced by the second-lowest bid.
- **Coalition Formation:** If a server is overloaded (i.e., $R > R_{\text{threshold}}$), it can form a coalition with another server, with load sharing determined using fairness metrics (like Shapley values*).

### 5. Learning and Updates
Servers adapt their bidding strategies based on auction outcomes:

#### Q-Learning Update:
  
$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
$$

- $\alpha$ is the learning rate.
- $\gamma$ is the discount factor.

#### Multi-Armed Bandit Update:

$$
V_a \leftarrow V_a + \frac{1}{n_a} \left( r - V_a \right)
$$

- $n_a$ is the number of times action $a$ has been selected.

### 6. Reserve Threshold Update
After each round, the auctioneer updates the reserve threshold based on the winning bids:

$$
T_{\text{reserve}} = 1.5 \times \text{Average Winning Bid}
$$

### 7. Metrics & Evaluation
The system collects metrics on:
- **Load Distribution:** Standard deviation of server loads.
- **Efficiency:** Task completion times.
- **Fairness:** Measured by load sharing among servers.


---
*Shapley Values in Coalition Formation

When a server is overloaded and forms a coalition with another server, Shapley values are used to fairly distribute the load based on each server's contribution. The Shapley value for a server $i$ is defined as:

$$
\phi_i = \frac{1}{n!}\sum_{\pi \in \Pi} \Bigl(v(S \cup \{i\}) - v(S)\Bigr)
$$

- $\phi_i$ is the Shapley value for server $i$.
- $\Pi$ is the set of all possible orderings of the servers.
- $S$ is the set of servers preceding server $i$ in a given ordering.
- $v(S)$ is the value (e.g., spare capacity) of coalition $S$.
- $n$ is the total number of servers.

This calculates the average marginal contribution of server $i$ by considering all possible ways the servers can form a coalition. By averaging the incremental contributions, the Shapley value provides a fair share of the load for each server in the coalition.
