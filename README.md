# Reinforcement, Shapley, Fair Allocation Load Balancing  
**Game-Theoretic Load Balancer**

This project implements a load balancer that uses auction-based task allocation combined with reinforcement learning and game-theoretic principles. Tasks are assigned to servers based on dynamically computed bids, and the system continuously adapts its parameters to balance loads efficiently.

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
  - **Capacity** ($C$) -  maximum workload
  - **Sensitivity** ($s$) - bid aggressiveness
  - **Risk Aversion** ($r$) - adjusts bids when near overload
- An **auctioneer** is set up to manage task assignments and update reserve thresholds.

### 2. Task Generation
- **Tasks** arrive in rounds.
- Each task has:
  - **Load Demand** ($L_t$) - the computational work needed to complete the task
  - **SLA Level** ($\lambda$) - different priorities of the tasks (silver, gold, bronze)
- Task arrival is dictated by the user (entered on the frontend, sent to the backend). The simulation will model sending tasks serially.

### 3. Bid Computation by Each Server
Each server computes its bid using several steps:

#### a. Calculate Load Ratio  
This ratio indicates how busy a server is:

$$
R = \min\left(\frac{L_{\text{current}}}{C}, 1\right)
$$

#### b. Compute Base Bid  
The base bid is computed by combining the server’s sensitivity, its load ratio, and the task's load:

$$
B_{\text{base}} = s \cdot R + L_t
$$

#### c. Apply Risk Adjustment  
To discourage overloaded servers from taking on additional tasks, a risk factor is applied:

$$
F = 1 + r \cdot \max\left(0, R - R_{\text{target}}\right)
$$

- $R_{\text{target}}$ - the ideal load a server should maintain
- If $R$ exceeds $R_{\text{target}}$, then the term $R - R_{\text{target}}$ becomes positive. The risk factor then increases linearly with the excess load, scaled by the risk aversion parameter: $r$
- This higher value of FF effectively increases the bid, making it less attractive for the server to take on more work.

The adjusted bid becomes:

$$
B_{\text{adj}} = B_{\text{base}} \cdot F
$$

#### d. Final Bid Scaling & Clamping  
The bid is then scaled by the server’s capacity and clamped to a maximum value:

$$
B_{\text{final}} = \min\left( B_{\text{adj}} \cdot \left(1 + \frac{C}{200}\right), 1000 \right)
$$

### 4. Auction Process
- **Reverse Auction:** - ALL servers submit bids, and the server with the lowest bid wins (because lower bid reflects less load, greater ability to take on tasks)
- **Second-Price Rule  / Vickrey Auction:** - the winning bidder pays the second lowest bid to determine truthful bidding (Can computer's deceptively bid when they're following the above formula?)
- **Coalition Formation:** - if a server is overloaded (i.e., $R > R_{\text{threshold}}$), it can form a coalition with another server; in such cases, fairness in load sharing is determined using Shapley values (Appendix).

### 5. Learning and Updates
Servers refine their bidding strategies using two approaches:

#### Q-Learning Update  
The Q-learning rule is applied as follows:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \Bigl( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \Bigr)
$$

- $\alpha$ is the learning rate.
- $\gamma$ is the discount factor.

#### Multi-Armed Bandit Update  
For a simpler update, the bandit method uses:

$$
V_a \leftarrow V_a + \frac{1}{n_a}\Bigl( r - V_a \Bigr)
$$

- $V_a$​ is the estimated value (expected reward) of bidding strategy $a$.
- $r$ is the received reward (e.g., successful task completion without exceeding capacity)
- $n_a$​ is the count of how many times bid $a$ has been selected.

### 6. Reserve Threshold Update
After each auction round, the auctioneer updates the reserve threshold based on winning bids:

$$
T_{\text{reserve}} = 1.5 \times \text{Average Winning Bid}
$$

### 7. Metrics & Evaluation
The system collects metrics on:
- **Load Distribution:**  - measured as the standard deviation of server loads
- **Efficiency:**  - total task completion times
- **Fairness:** -  evaluated by how load is shared among servers

---

## Appendix: Shapley Values in Coalition Formation

When a server is overloaded and forms a coalition with another server, **Shapley values** ensure that the load is shared fairly based on each server's contribution. The Shapley value for server $i$ is defined as:

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\, (|N|-|S|-1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]
$$

Where:
- $N$ is the set of all servers.
- $S$ is any subset of servers excluding server $i$.
- $v(S)$ is the value (e.g., available capacity or cost efficiency) of coalition $S$.
- The fraction is a weight that accounts for the number of possible orders in which the coalition could be formed.
- 
This calculates the average marginal contribution of server $i$ over all possible coalitions. It ensures that each server’s share of the load is proportional to its individual contribution, leading to a fair distribution when servers collaborate.
