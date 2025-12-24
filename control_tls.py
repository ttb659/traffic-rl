import traci
import time

sumo_cmd = ["sumo-gui", "-c", "sumo/simulation.sumocfg"]
traci.start(sumo_cmd)

tls_ids = traci.trafficlight.getIDList()
print("Feux détectés :", tls_ids)

if len(tls_ids) == 0:
    print("❌ Aucun feu détecté. Arrêt.")
    traci.close()
    exit()

tls_id = tls_ids[0]
print("Feu contrôlé :", tls_id)

# 🔎 Récupération des phases correctement
programs = traci.trafficlight.getAllProgramLogics(tls_id)
phases = programs[0].phases
num_phases = len(phases)

print("Nombre de phases :", num_phases)

for step in range(200):
    traci.simulationStep()

    phase = traci.trafficlight.getPhase(tls_id)
    print(f"Step {step} | Phase actuelle : {phase}")

    # Changement toutes les 40 étapes
    if step % 40 == 0:
        new_phase = (phase + 1) % num_phases
        traci.trafficlight.setPhase(tls_id, new_phase)
        print(" → Changement vers phase", new_phase)

    time.sleep(0.05)

traci.close()
