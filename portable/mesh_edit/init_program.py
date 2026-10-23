"""Initial weak local smoothing heuristic. Preserve the public solve API."""
# EVOLVE-BLOCK-START
def solve(scene):
    import numpy as np
    v = np.asarray(scene['vertices'], dtype=float)
    f = np.asarray(scene['faces'], dtype=int)
    editable = np.asarray(scene['editable'], dtype=bool)
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    current = v.copy()
    for _ in range(2):
        sums = np.zeros_like(current)
        degree = np.zeros(len(current))
        for a, b in ((0, 1), (1, 0)):
            np.add.at(sums, edges[:, a], current[edges[:, b]])
            np.add.at(degree, edges[:, a], 1)
        target = sums / np.maximum(degree[:, None], 1)
        current[editable] = .65*current[editable]+.35*target[editable]
    current[~editable] = v[~editable]
    return {'vertices': current.tolist()}
# EVOLVE-BLOCK-END
