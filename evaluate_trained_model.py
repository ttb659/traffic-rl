import torch
import numpy as np
import time

from env.multi_agent_traffic_env import MultiAgentTrafficEnv
from gnn.light_gnn import LightGNN
from marl.actor import Actor
from marl.critic import Critic
from marl.mappo import MAPPO
from graph.adjacency import build_adjacency_matrix
from graph.normalize import normalize_adjacency
from graph.mapping import AGENT_IDS

# =====================
# CONFIG
# =====================
SUMO_CFG = "sumo/simulation_3x3.sumocfg"
CHECKPOINT_PATH = "models_save/mappo_final+Light_Gnn.pt"
EP_LEN = 3600          # durée de la démo
USE_GUI = True

OBS_DIM = 6
EMB_DIM = 32
N_AGENTS = 9
ACTION_DIM = 2

# =====================
# ENV (GUI)
# =====================
env = MultiAgentTrafficEnv(
    SUMO_CFG,
    max_steps=EP_LEN,
    use_gui=USE_GUI
)

# =====================
# GRAPH
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

mappo = MAPPO(actor, critic, gnn)

# =====================
# LOAD CHECKPOINT
# =====================
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
actor.load_state_dict(checkpoint["actor"])
critic.load_state_dict(checkpoint["critic"])
gnn.load_state_dict(checkpoint["gnn"])

actor.eval()
critic.eval()
gnn.eval()

print("Modèle chargé avec succès")

# =====================
# RUN DEMO
# =====================
obs = env.reset()
total_queue = 0

for t in range(EP_LEN):
    state = torch.tensor(
        np.array([obs[aid] for aid in AGENT_IDS]),
        dtype=torch.float32
    )

    with torch.no_grad():
        embeddings = gnn(state, A_norm)
        dist = actor(embeddings)

        # action déterministe (meilleure action)
        actions = torch.argmax(dist.probs, dim=1)

    action_dict = {
        aid: actions[i].item()
        for i, aid in enumerate(AGENT_IDS)
    }

    obs, rewards, dones = env.step(action_dict)

    step_queue = sum(-r for r in rewards.values())
    total_queue += step_queue

    # ralentir pour visualisation humaine
    time.sleep(0.05)

    if dones["__all__"]:
        break

avg_queue = total_queue / t
print(f"Avg queue during demo = {avg_queue:.2f}")

env.close()
