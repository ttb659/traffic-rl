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
# PARAMÈTRES
# =====================
OBS_DIM = 6
EMB_DIM = 32
N_AGENTS = 9
ACTION_DIM = 2

EPISODES = 50        # commence petit
EP_LEN = 200

# =====================
# ENV
# =====================
env = MultiAgentTrafficEnv(
    "sumo/simulation_3x3.sumocfg",
    max_steps=EP_LEN
)

# =====================
# GRAPHE
# =====================
A = build_adjacency_matrix()
A_norm = torch.tensor(
    normalize_adjacency(A),
    dtype=torch.float32
)

# =====================
# MODELS
# =====================
gnn = LightGNN(OBS_DIM, 32, EMB_DIM)
actor = Actor(EMB_DIM, ACTION_DIM)
critic = Critic(EMB_DIM, N_AGENTS)

mappo = MAPPO(actor, critic, gnn, lr=3e-4)

# =====================
# TRAINING
# =====================
for episode in range(EPISODES):
    obs = env.reset()

    states, actions, log_probs, rewards = [], [], [], []
    ep_reward = 0.0

    for t in range(EP_LEN):
        state = torch.tensor(
            np.array([obs[aid] for aid in AGENT_IDS]),
            dtype=torch.float32
        )

        with torch.no_grad():
            embeddings = gnn(state, A_norm)
            dist = actor(embeddings)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        action_dict = {
            aid: action[i].item()
            for i, aid in enumerate(AGENT_IDS)
        }

        obs, reward_dict, dones = env.step(action_dict)

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)

        total_reward = sum(reward_dict.values())
        rewards.append(total_reward)
        ep_reward += total_reward

        if dones["__all__"]:
            break

    batch = {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "log_probs": torch.stack(log_probs),
        "rewards": rewards,
        "adj": A_norm
    }

    mappo.update(batch)

    if episode % 5 == 0:
        print(f"[Episode {episode}] Total reward = {ep_reward:.2f}")

env.close()
