import numpy as np
data = np.array([
    [1, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1]
])
vigilance = 0.8
clusters = []
for pattern in data:
    scores = [
        np.sum(c * pattern) / (0.01 + np.sum(c))
        for c in clusters
    ]
    for i in np.argsort(scores)[::-1]:
        match = np.sum(pattern * clusters[i]) / np.sum(pattern)
        if match >= vigilance:
            clusters[i] = clusters[i] * pattern
            break
    else:
        clusters.append(pattern.copy())
        i = len(clusters) - 1
    print(pattern, "-> Cluster", i + 1)
print("\nFinal Prototypes:")
for i, c in enumerate(clusters):
    print("Cluster", i + 1, ":", c)
