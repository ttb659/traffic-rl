from baselines.fixed_tls import run_fixed_tls

avg_queue = run_fixed_tls(
    "sumo/simulation_3x3_fixed.sumocfg",
    steps=200
)

print(f"[FIXED TLS] Avg Queue = {avg_queue:.2f}")
