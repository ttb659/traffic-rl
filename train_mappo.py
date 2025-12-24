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
from utils.logger import Logger



# =====================
# PARAMÈTRES
# =====================
OBS_DIM = 6
EMB_DIM = 32
N_AGENTS = 9
ACTION_DIM = 2

EPISODES =50      # commence petit
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
USE_GNN = False  # <-- ABLATION ICI POUR TESTER SANS GNN

if USE_GNN:
    logger = Logger()
    from gnn.light_gnn import LightGNN
    gnn = LightGNN(OBS_DIM, 32, EMB_DIM)
    embed_dim = EMB_DIM
else:
    logger = Logger(filename="train_log_no_gnn.csv")
    from gnn.identity_gnn import IdentityGNN
    gnn = IdentityGNN()
    embed_dim = OBS_DIM


actor = Actor(embed_dim, ACTION_DIM)
critic = Critic(embed_dim, N_AGENTS)

mappo = MAPPO(actor, critic, gnn, lr=1e-4)

# =====================
# TRAINING
# =====================
for episode in range(EPISODES):
    obs = env.reset()

    states, actions, log_probs, rewards = [], [], [], []
    ep_reward = 0.0

    total_queue = 0

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
        step_queue = sum(reward_dict.values()) * -1  # waiting time positif
        total_queue += step_queue


        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)

        total_reward = sum(reward_dict.values()) / N_AGENTS
        rewards.append(total_reward)
        ep_reward += total_reward

        if dones["__all__"]:
            break
    
    avg_queue = total_queue / EP_LEN
    avg_reward_per_step = ep_reward / EP_LEN


    batch = {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "log_probs": torch.stack(log_probs),
        "rewards": rewards,
        "adj": A_norm
    }

    mappo.update(batch)
    logger.log([
        episode,
        ep_reward,
        avg_reward_per_step,
        mappo.last_actor_loss,
        mappo.last_critic_loss,
        mappo.last_entropy,
        avg_queue
    ])


    if episode % 5 == 0:
        print(
            f"[Episode {episode:03d}] "
            f"Reward: {ep_reward:8.1f} | "
            f"Actor: {mappo.last_actor_loss:6.3f} | "
            f"Critic: {mappo.last_critic_loss:6.3f} | "
            f"Entropy: {mappo.last_entropy:5.3f} | "
            f"AvgQueue: {avg_queue:6.1f}"
        )

env.close()
