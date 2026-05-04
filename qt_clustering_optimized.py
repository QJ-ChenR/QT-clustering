#!/usr/bin/env python3
import math
import sys
import os


class QTClustering:
    """
    Correct Quality Threshold Clustering
    """

    def __init__(self, points, threshold):
        self.points = points
        self.threshold = threshold
        self.n = len(points)

        self.dist_matrix = self._compute_distance_matrix()
        self.neighbors = self._build_neighbor_matrix()

        # Track unclustered points
        self.active = [True] * self.n

    # --------------------------------------------------
    # Distance Functions
    # --------------------------------------------------

    def _euclidean_distance(self, p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def _compute_distance_matrix(self):
        matrix = [[0.0] * self.n for _ in range(self.n)]

        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = self._euclidean_distance(self.points[i], self.points[j])
                matrix[i][j] = d
                matrix[j][i] = d

        return matrix

    # --------------------------------------------------
    # Neighbor Matrix
    # --------------------------------------------------

    def _build_neighbor_matrix(self):
        neighbors = []

        for i in range(self.n):
            row = set()

            for j in range(self.n):
                if i != j and self.dist_matrix[i][j] <= self.threshold:
                    row.add(j)

            neighbors.append(row)

        return neighbors

    # --------------------------------------------------
    # Cluster Utilities
    # --------------------------------------------------

    def cluster_diameter(self, cluster):
        max_d = 0.0

        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                d = self.dist_matrix[cluster[i]][cluster[j]]
                if d > max_d:
                    max_d = d

        return max_d

    # --------------------------------------------------
    # Candidate Cluster Construction
    # --------------------------------------------------

    def build_candidate_cluster(self, seed):
        """
        Standard QT candidate construction:
        - Start from seed
        - Iteratively add point preserving full clique
        - Greedy choose point maximizing future candidate set
        """

        cluster = [seed]

        candidates = {
            p for p in self.neighbors[seed]
            if self.active[p]
        }

        while candidates:

            best_point = None
            best_future_candidates = set()

            for p in candidates:

                # Remaining candidates after adding p
                future_candidates = {
                    q for q in candidates
                    if q != p
                    and self.active[q]
                    and self.dist_matrix[p][q] <= self.threshold
                }

                # Choose candidate preserving largest future set
                if len(future_candidates) > len(best_future_candidates):
                    best_point = p
                    best_future_candidates = future_candidates

            if best_point is None:
                break

            # Add best point
            cluster.append(best_point)

            # Shrink candidates
            candidates = {
                q for q in best_future_candidates
                if all(
                    self.dist_matrix[q][member] <= self.threshold
                    for member in cluster
                )
            }

        return cluster

    # --------------------------------------------------
    # Main QT Clustering
    # --------------------------------------------------

    def fit(self):
        clusters = []

        remaining_points = sum(self.active)

        while remaining_points > 0:

            best_cluster = None
            best_size = 0
            best_diameter = float("inf")

            for seed in range(self.n):

                if not self.active[seed]:
                    continue

                # Count only active neighbors
                active_neighbors = sum(
                    1 for p in self.neighbors[seed]
                    if self.active[p]
                )

                # Seed pruning
                if active_neighbors + 1 < best_size:
                    continue

                candidate = self.build_candidate_cluster(seed)

                candidate_size = len(candidate)
                candidate_diameter = self.cluster_diameter(candidate)

                # Standard QT tie-breaking
                if (
                    candidate_size > best_size
                    or (
                        candidate_size == best_size
                        and candidate_diameter < best_diameter
                    )
                ):
                    best_cluster = candidate
                    best_size = candidate_size
                    best_diameter = candidate_diameter

            # Safety fallback
            if best_cluster is None:
                break

            clusters.append(best_cluster)

            # Deactivate clustered points
            for idx in best_cluster:
                if self.active[idx]:
                    self.active[idx] = False
                    remaining_points -= 1

        return clusters


# --------------------------------------------------
# File Utilities
# --------------------------------------------------

def load_points_from_file(filename):
    points = []
    labels = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            cols = line.split()

            labels.append(cols[0])
            points.append(tuple(float(x) for x in cols[1:]))

    return points, labels


def compute_max_distance(points):
    max_dist = 0.0
    n = len(points)

    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(
                sum(
                    (points[i][k] - points[j][k]) ** 2
                    for k in range(len(points[i]))
                )
            )

            if d > max_dist:
                max_dist = d

    return max_dist


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python qt_clustering.py <inputfile> <threshold>")
        print("Threshold can be absolute (e.g. 0.5) or percentage (e.g. 30%)")
        sys.exit(1)

    input_file = sys.argv[1]
    threshold_arg = sys.argv[2]

    if not os.path.exists(input_file):
        print("File not found:", input_file)
        sys.exit(1)

    points, labels = load_points_from_file(input_file)

    # Threshold parsing
    if threshold_arg.endswith("%"):
        percentage = float(threshold_arg[:-1]) / 100.0
        max_dist = compute_max_distance(points)
        threshold = max_dist * percentage
    else:
        threshold = float(threshold_arg)

    # Run clustering
    qt = QTClustering(points, threshold)
    clusters = qt.fit()

    # Output
    print("Threshold:", threshold)
    print("Total clusters:", len(clusters))

    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1} size: {len(cluster)}")
        print("Members:", [labels[idx] for idx in cluster])