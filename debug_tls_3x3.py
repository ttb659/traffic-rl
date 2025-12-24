import traci

traci.start(["sumo", "-c", "sumo/simulation_3x3.sumocfg"])

tls_ids = traci.trafficlight.getIDList()
print("Nombre de feux détectés :", len(tls_ids))
print("IDs :", tls_ids)

traci.close()
