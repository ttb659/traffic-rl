import traci
import numpy as np
import time

from graph.mapping import AGENT_IDS, AGENT_TO_IDX


class MultiAgentTrafficEnv:
    def __init__(self, sumo_config, max_steps=1000):
        self.sumo_config = sumo_config
        self.max_steps = max_steps
        self.current_step = 0

        self.agent_ids = AGENT_IDS
        self.n_agents = len(self.agent_ids)

        self.last_switch_time = {aid: 0 for aid in self.agent_ids}

    # ==============================
    # RESET
    # ==============================
    def reset(self):
        if traci.isLoaded():
            traci.close()

        traci.start(["sumo", "-c", self.sumo_config])
        self.current_step = 0

        # Vérification agents
        tls_ids = traci.trafficlight.getIDList()
        for aid in self.agent_ids:
            assert aid in tls_ids, f"Feu {aid} introuvable dans SUMO"

        return self._get_observations()

    # ==============================
    # STEP
    # ==============================
    def step(self, actions):
        """
        actions : dict {agent_id: action}
        """
        self.current_step += 1

        # Appliquer actions
        for aid, action in actions.items():
            if action == 1:
                self._switch_phase(aid)

        traci.simulationStep()

        observations = self._get_observations()
        rewards = self._get_rewards()
        done = self.current_step >= self.max_steps

        dones = {aid: done for aid in self.agent_ids}
        dones["__all__"] = done

        return observations, rewards, dones

    # ==============================
    # OBSERVATIONS
    # ==============================
    def _get_observations(self):
        obs = {}

        for aid in self.agent_ids:
            lanes = traci.trafficlight.getControlledLanes(aid)

            # Comptage voitures arrêtées
            halting = [traci.lane.getLastStepHaltingNumber(l) for l in lanes]

            # Pad / trim à 4 directions
            halting = halting[:4] + [0] * max(0, 4 - len(halting))

            phase = traci.trafficlight.getPhase(aid)
            time_since_switch = self.current_step - self.last_switch_time[aid]

            obs[aid] = np.array(
                halting + [phase, time_since_switch],
                dtype=np.float32
            )

        return obs

    # ==============================
    # REWARD
    # ==============================
    def _get_rewards(self):
        rewards = {}

        for aid in self.agent_ids:
            lanes = traci.trafficlight.getControlledLanes(aid)
            waiting = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)

            # Récompense locale (négative)
            rewards[aid] = -float(waiting)

        return rewards

    # ==============================
    # PHASE SWITCH
    # ==============================
    def _switch_phase(self, aid):
        programs = traci.trafficlight.getAllProgramLogics(aid)
        phases = programs[0].phases
        num_phases = len(phases)

        current_phase = traci.trafficlight.getPhase(aid)
        new_phase = (current_phase + 1) % num_phases

        traci.trafficlight.setPhase(aid, new_phase)
        self.last_switch_time[aid] = self.current_step

    # ==============================
    # CLOSE
    # ==============================
    def close(self):
        traci.close()
