"""
1. calculate the euclidian distance between two points
2. sort the points by distance
3. return the k nearest points
"""

import numpy as np


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2) ** 2))


def knn(points: np.ndarray, k: int) -> list:
    """
    Find k nearest neighbors to the first point in the array.

    Args:
        points: Array of points where each row is a point
        k: Number of nearest neighbors to find

    Returns:
        List of indices of k nearest neighbors to the first point (including the first point)
    """
    if len(points) == 0 or k <= 0:
        return []

    # Use the first point as the query point
    query_point = points[0]

    # Calculate distances from query point to all other points
    distances = []
    for i in range(len(points)):
        distance = euclidean_distance(query_point, points[i])
        distances.append((distance, i))

    # Sort by distance and get the k nearest neighbors (including the query point itself)
    distances.sort()

    # Return indices of k nearest neighbors (including the query point)
    nearest_indices = [idx for _, idx in distances[:k]]

    return nearest_indices
