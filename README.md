# 🚦 Multi-Agent Reinforcement Learning for Traffic Signal Control (SUMO)

## 📌 Project Overview

This project implements a **Multi-Agent Reinforcement Learning (MARL)** approach for **urban traffic signal control** using **SUMO**.
Each traffic light is modeled as an autonomous agent, and coordination between intersections is achieved using **MAPPO (Multi-Agent PPO)** with an optional **Light Graph Neural Network (Light-GNN)** for spatial information sharing.

The project is designed as an **academic-grade implementation**, suitable for:

* Master / engineering thesis
* Research-oriented projects
* Portfolio / CV demonstration in Reinforcement Learning and Intelligent Transportation Systems

---

## 🎯 Objectives

* Model a **3×3 urban road network** in SUMO
* Control **9 traffic lights (1 agent per intersection)**
* Implement **MAPPO from scratch** (no RLlib, no Stable-Baselines)
* Integrate a **Light-GNN** for spatial coordination
* Compare against **classical baselines**:

  * Fixed-time traffic lights
  * SUMO actuated traffic lights
* Perform a **clean ablation study** (with vs without GNN)

---

## 🧠 Final Architecture

```
SUMO (3×3 intersections)
│
├── Local Observations (per traffic light)
│     ├─ Queue length per direction (N/S/E/W)
│     ├─ Current signal phase
│     └─ Time since last phase switch
│
├── Static Road Graph
│     └─ Adjacency matrix (direct neighbors)
│
├── Light-GNN (optional, shared parameters)
│     └─ Spatial state encoding
│
└── MAPPO (Actor–Critic)
      ├─ Decentralized Actors (shared policy)
      └─ Centralized Critic (global value function)
```

**Training paradigm**: Centralized Training & Decentralized Execution (CTDE)

---

## 🏗️ Project Structure

```
traffic-rl/
├── agents/              # Actor & Critic networks
├── marl/                # MAPPO implementation
├── gnn/                 # Light-GNN & IdentityGNN (ablation)
├── graph/               # Graph topology & normalization
├── env/                 # Multi-agent SUMO environment
├── sumo/                # SUMO network, routes & configs
├── baselines/           # Fixed-time & actuated baselines
├── logs/                # Training logs (CSV)
├── train_mappo.py       # Main training script
├── analyze_results.py   # Metrics analysis
├── plot_comparison.py   # Bar chart comparison
├── plot_learning_curves.py
└── README.md
```

---

## 🚦 Environment Details

* **Simulator**: SUMO (via TraCI)
* **Network**: 3×3 grid (9 intersections)
* **Agents**: 9 traffic lights
* **Action space**:

  * `0`: keep current phase
  * `1`: switch to next phase
* **Reward**:

  * Negative global waiting time (cooperative reward)
  * Normalized by number of agents

---

## 🤖 Learning Algorithms

### 🔹 MAPPO (Multi-Agent PPO)

* Shared actor parameters
* Centralized critic using joint agent embeddings
* Generalized Advantage Estimation (GAE)
* PPO clipped objective
* CPU-friendly implementation

### 🔹 Light-GNN (Optional)

* Static graph (Manhattan grid)
* Lightweight message passing (GCN-style)
* Improves spatial coordination between intersections

### 🔹 Ablation

* **MAPPO + GNN**
* **MAPPO without GNN** (IdentityGNN)

---

## 📊 Baselines

| Method         | Description                       |
| -------------- | --------------------------------- |
| Fixed TLS      | No learning, static signal phases |
| Actuated TLS   | SUMO built-in actuated control    |
| MAPPO (no GNN) | MARL with local observations only |
| MAPPO + GNN    | Full proposed method ⭐            |

---

## 📈 Results Summary

Key metric: **Average Queue Length (lower is better)**

* Reinforcement learning approaches significantly outperform classical traffic control strategies.
* Fixed-time control can outperform actuated control in small, dense networks due to reduced oscillations.
* The Light-GNN introduces higher variance but enables richer spatial coordination and better scalability.

> Full results are available in the `logs/` directory and visualized using the provided plotting scripts.

---

## 🧪 How to Run

### 1️⃣ Requirements

* Python ≥ 3.8
* SUMO (with TraCI)
* PyTorch (CPU version)

```bash
pip install torch numpy pandas matplotlib
```

---

### 2️⃣ Training MAPPO + GNN

```bash
python train_mappo.py
```

To disable the GNN (ablation):

```python
USE_GNN = False
```

---

### 3️⃣ Run Baselines

```bash
python run_fixed_baseline.py
python run_actuated_baseline.py
```

---

### 4️⃣ Plot Results

```bash
python analyze_results.py
python plot_comparison.py
python plot_learning_curves.py
```

---

## 📚 References

* Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2021
* Wei et al., *CoLight: Learning Network-Level Cooperation for Traffic Signal Control*, AAAI
* PressLight, KDD
* Graph-based MARL for Traffic Signal Control, IEEE T-ITS

---

## 🎓 Key Takeaways

* Demonstrates a **full MARL pipeline** with SUMO
* Clean **from-scratch MAPPO implementation**
* Proper **ablation study and baselines**
* Designed for **academic rigor and reproducibility**

---

## 👤 Author

**TIDO TAMEKENG BOREL**
Project developed for academic and research purposes.

---

## ⭐ Final Note

If you find this project useful, feel free to ⭐ the repository or use it as a reference for research and learning purposes.

