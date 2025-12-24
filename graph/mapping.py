AGENT_IDS = [
    "B1", "B2", "B3",
    "C1", "C2", "C3",
    "D1", "D2", "D3"
]

# agent_id -> node index (0..8)
AGENT_TO_IDX = {agent: i for i, agent in enumerate(AGENT_IDS)}

# node index -> grid position (row, col)
IDX_TO_POS = {
    0: (0, 0), 1: (0, 1), 2: (0, 2),
    3: (1, 0), 4: (1, 1), 5: (1, 2),
    6: (2, 0), 7: (2, 1), 8: (2, 2),
}
