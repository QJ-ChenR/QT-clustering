import os
import sys
import pytest
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# This line is added to ensure that the qt_clustering module can be imported from the parent directory, where it is assumed to be located.

from qt_clustering_optimized import QTClusterer, load_points, _euclidean_distance


TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "../testdata")


def load_dataset(filename):
    path = os.path.join(TESTDATA_DIR, filename)
    data = load_points(path)
    labels = [label for label, _ in data]
    points = [coords for _, coords in data]
    return labels, points

# -------------------------
# Core correctness tests
# -------------------------

@pytest.mark.parametrize("filename", ["point100.lst", "point1000.lst"])
def test_all_points_assigned(filename):
    labels, points = load_dataset(filename)

    clusterer = QTClusterer(threshold=30)
    clusters = clusterer.fit(points)

    assigned = sorted(idx for cluster in clusters for idx in cluster)

    assert assigned == list(range(len(points)))


def test_no_duplicate_assignment():
    # This test ensures that no point is assigned to more than one cluster.
    labels, points = load_dataset("point100.lst")

    clusterer = QTClusterer(threshold=30)
    clusters = clusterer.fit(points)

    seen = set()
    for cluster in clusters:
        for idx in cluster:
            assert idx not in seen
            seen.add(idx)


def test_deterministic():
    labels, points = load_dataset("point100.lst")

    clusterer = QTClusterer(threshold=30)

    c1 = clusterer.fit(points)
    c2 = clusterer.fit(points)

    assert c1 == c2

# -------------------------
# Mathematical correctness
# -------------------------

def test_euclidean_distance_symmetry():
    a = (0.0, 0.0)
    b = (3.0, 4.0)

    assert _euclidean_distance(a, b) == _euclidean_distance(b, a)
    assert math.isclose(_euclidean_distance(a, b), 5.0)


def test_distance_matrix_symmetry():
    points = [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]

    clusterer = QTClusterer(threshold=10)
    matrix = clusterer._distance_matrix(points)

    for i in range(len(points)):
        for j in range(len(points)):
            assert math.isclose(matrix[i][j], matrix[j][i])


# -------------------------
# Input validation tests
# -------------------------

def test_invalid_threshold():
    with pytest.raises(ValueError):
        QTClusterer(threshold=-1)


def test_dimension_mismatch():
    points = [
        (1.0, 2.0),
        (3.0, 4.0, 5.0),
    ]

    clusterer = QTClusterer(threshold=10)

    with pytest.raises(ValueError):
        clusterer.fit(points)


def test_empty_points():
    clusterer = QTClusterer(threshold=10)
    clusters = clusterer.fit([])

    assert clusters == []


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_points("nonexistent_file.lst")


def test_malformed_point_line(tmp_path):
    malformed_file = tmp_path / "bad.lst"
    malformed_file.write_text("badline\n")

    with pytest.raises(ValueError):
        load_points(str(malformed_file))


def test_non_numeric_coordinates(tmp_path):
    malformed_file = tmp_path / "bad_coords.lst"
    malformed_file.write_text("p1 a b c\n")

    with pytest.raises(ValueError):
        load_points(str(malformed_file))


def test_declared_count_mismatch(tmp_path):
    bad_file = tmp_path / "count_mismatch.lst"
    bad_file.write_text("2\np1 1 2\n")

    with pytest.raises(ValueError):
        load_points(str(bad_file))


# -------------------------
# Neighbor pruning
# -------------------------

def test_neighbor_list_threshold():
    points = [
        (0.0, 0.0),
        (1.0, 1.0),
        (10.0, 10.0),
    ]

    clusterer = QTClusterer(threshold=2)
    distances = clusterer._distance_matrix(points)
    neighbors = clusterer._build_neighbor_lists(distances, 2)

    assert 1 in neighbors[0]
    assert 2 not in neighbors[0]


# -------------------------
# Small exact clustering test
# -------------------------

def test_small_known_clustering():
    points = [
        (0.0, 0.0),
        (1.0, 1.0),
        (10.0, 10.0),
        (11.0, 11.0),
    ]

    clusterer = QTClusterer(threshold=3)
    clusters = clusterer.fit(points)

    cluster_sizes = sorted(len(cluster) for cluster in clusters)

    assert cluster_sizes == [2, 2]

