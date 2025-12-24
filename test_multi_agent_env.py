from env.multi_agent_traffic_env import MultiAgentTrafficEnv
import random
import time

env = MultiAgentTrafficEnv("sumo/simulation_3x3.sumocfg", max_steps=200)
obs = env.reset()

print("Observation initiale d’un agent :", obs["B1"])

for step in range(50):
    actions = {aid: random.randint(0, 1) for aid in obs.keys()}
    obs, rewards, dones = env.step(actions)

    print(f"Step {step} | Reward B1 = {rewards['B1']}")

    if dones["__all__"]:
        break

    time.sleep(0.1)

env.close()
