from env.traffic_env import TrafficEnv
import time

env = TrafficEnv("sumo/simulation.sumocfg")
state = env.reset()

print("État initial :", state)

for step in range(50):
    action = step % 2  # alterne actions
    state, reward, done = env.step(action)
    print(f"Step {step} | Action {action} | Reward {reward} | State {state}")
    time.sleep(0.1)

env.close()
