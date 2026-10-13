import numpy as np
data = np.array([
    [0.9, 0.1, 0.2],
    [0.8, 0.2, 0.1],
    [0.1, 0.9, 0.2],
    [0.2, 0.8, 0.1]
], dtype=float)
vigilance = 0.9
clusters = []
for pattern in data:
    norm = np.linalg.norm(pattern)
    pattern = pattern / norm if norm != 0 else pattern
    assigned = False
    for i, c in enumerate(clusters):
        similarity = np.dot(pattern, c) / (
            np.linalg.norm(c) + 1e-10
        )
        if similarity >= vigilance:
            clusters[i] = (clusters[i] + pattern) / 2
            clusters[i] /= np.linalg.norm(clusters[i]) + 1e-10
            print(pattern, "-> Cluster", i + 1)
            assigned = True
            break
    if not assigned:
        clusters.append(pattern.copy())
        print(pattern, "-> Cluster", len(clusters))
print("\nFinal ART2 Prototypes:")
for i, c in enumerate(clusters):
    print("Cluster", i + 1, ":", np.round(c, 3))
