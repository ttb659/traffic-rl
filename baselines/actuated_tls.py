import traci
from graph.mapping import AGENT_IDS


def run_actuated_tls(sumo_cfg, steps=200):
    traci.start(["sumo", "-c", sumo_cfg])

    total_queue = 0

    for step in range(steps):
        traci.simulationStep()

        step_queue = 0
        for aid in AGENT_IDS:
            lanes = traci.trafficlight.getControlledLanes(aid)
            step_queue += sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)

        total_queue += step_queue

    traci.close()

    avg_queue = total_queue / steps
    return avg_queue
