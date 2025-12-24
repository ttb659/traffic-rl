from baselines.actuated_tls import run_actuated_tls

avg_queue = run_actuated_tls(
    "sumo/simulation_3x3.sumocfg",
    steps=200
)

print(f"[ACTUATED TLS] Avg Queue = {avg_queue:.2f}")
