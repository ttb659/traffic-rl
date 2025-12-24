import traci
import numpy as np

class TrafficEnv:
    def __init__(self, sumo_config):
        self.sumo_config = sumo_config
        self.tls_id = None

    def reset(self):
        traci.start(["sumo", "-c", self.sumo_config])

        self.tls_id = traci.trafficlight.getIDList()[0]
        traci.simulationStep()

        return self._get_state()

    def step(self, action):
        # Action: 0 = rien, 1 = changer de phase
        if action == 1:
            programs = traci.trafficlight.getAllProgramLogics(self.tls_id)
            num_phases = len(programs[0].phases)
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            new_phase = (current_phase + 1) % num_phases
            traci.trafficlight.setPhase(self.tls_id, new_phase)

        traci.simulationStep()

        state = self._get_state()
        reward = -self._get_total_waiting_cars()
        done = False  # épisode continu

        return state, reward, done

    def _get_state(self):
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        halting = [traci.lane.getLastStepHaltingNumber(l) for l in lanes]
        phase = traci.trafficlight.getPhase(self.tls_id)

        return np.array(halting + [phase], dtype=np.float32)

    def _get_total_waiting_cars(self):
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        return sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)

    def close(self):
        traci.close()
