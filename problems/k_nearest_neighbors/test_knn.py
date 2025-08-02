from knn import knn
import numpy as np
import pytest


@pytest.mark.parametrize(
    "points, k, expected",
    [
        (np.array([[0, 0], [1, 1], [2, 2], [3, 3]]), 2, [0, 1]),
        (np.array([[0, 0], [1, 1], [2, 2], [3, 3]]), 3, [0, 1, 2]),
        (np.array([[0, 0], [1, 1], [2, 2], [3, 3]]), 4, [0, 1, 2, 3]),
    ],
)
def test_knn(points, k, expected):
    assert knn(points, k) == expected
