import torch
import numpy as np

from env.multi_agent_traffic_env import MultiAgentTrafficEnv
from gnn.light_gnn import LightGNN
from marl.actor import Actor
from marl.critic import Critic
from marl.mappo import MAPPO
from graph.adjacency import build_adjacency_matrix
from graph.normalize import normalize_adjacency
from graph.mapping import AGENT_IDS

# =====================
# PARAMÈTRES DEBUG
# =====================
OBS_DIM = 6
EMB_DIM = 32
N_AGENTS = 9
ACTION_DIM = 2

EPISODES = 3        # très court
EP_LEN = 50         # très court

# =====================
# ENVIRONNEMENT
# =====================
env = MultiAgentTrafficEnv(
    "sumo/simulation_3x3.sumocfg",
    max_steps=EP_LEN
)

# =====================
# GRAPHE
# =====================
A = build_adjacency_matrix()
A_norm = normalize_adjacency(A)
A_norm = torch.tensor(A_norm, dtype=torch.float32)

# =====================
# MODÈLES
# =====================
gnn = LightGNN(OBS_DIM, 32, EMB_DIM)
actor = Actor(EMB_DIM, ACTION_DIM)
critic = Critic(EMB_DIM, N_AGENTS)

mappo = MAPPO(actor, critic, gnn, lr=3e-4)

# =====================
# TRAINING DEBUG
# =====================
for episode in range(EPISODES):
    obs = env.reset()

    states = []
    actions = []
    log_probs = []
    rewards = []

    ep_reward = 0.0

    for t in range(EP_LEN):
        # ---------
        # Préparer état (N, obs_dim)
        # ---------
        state = torch.tensor(
            np.array([obs[aid] for aid in AGENT_IDS]),
            dtype=torch.float32
        )

        # ---------
        # GNN
        # ---------
        with torch.no_grad():
            embeddings = gnn(state, A_norm)

        # ---------
        # Actor
        # ---------
        dist = actor(embeddings)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # ---------
        # Step environnement
        # ---------
        action_dict = {
            aid: action[i].item()
            for i, aid in enumerate(AGENT_IDS)
        }

        obs, reward_dict, dones = env.step(action_dict)

        # ---------
        # Stockage
        # ---------
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)

        total_reward = sum(reward_dict.values())
        rewards.append(total_reward)
        ep_reward += total_reward

        if dones["__all__"]:
            break

    # =====================
    # UPDATE MAPPO
    # =====================
    batch = {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "log_probs": torch.stack(log_probs),
        "rewards": rewards,
        "adj": A_norm
    }

    mappo.update(batch)

    print(f"[Episode {episode}] Reward total = {ep_reward:.2f}")

env.close()
