import traci
import time

sumo_cmd = [
    "sumo-gui",
    "-c", "sumo/simulation.sumocfg"
    #,"--start"  # <-- Démarre automatiquement la simulation
]

print("Démarrage de SUMO...")
traci.start(sumo_cmd)

for step in range(200):
    traci.simulationStep()
    veh_ids = traci.vehicle.getIDList()
    print(f"Step {step} | Nombre de véhicules : {len(veh_ids)}")
    time.sleep(0.05)

traci.close()
print("Simulation terminée.")
