import os
import sys
import pytest

sys.path.append('..')

from qt_clustering import QTClusterer, load_points


TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "../testdata")


def load_dataset(filename):
    path = os.path.join(TESTDATA_DIR, filename)
    data = load_points(path)
    labels = [label for label, _ in data]
    points = [coords for _, coords in data]
    return labels, points


@pytest.mark.parametrize("filename", ["point100.lst", "point3000.lst"])
def test_all_points_assigned(filename):
    labels, points = load_dataset(filename)

    clusterer = QTClusterer(threshold=30)
    clusters = clusterer.fit(points)

    assigned = sorted(idx for cluster in clusters for idx in cluster)

    assert assigned == list(range(len(points)))


def test_no_duplicate_assignment():
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