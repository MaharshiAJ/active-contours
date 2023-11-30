import numpy as np


class ContourPoint:
    def __init__(self, x, y, next=None, previous=None):
        self.x = x
        self.y = y
        self.next = next
        self.previous = previous

    def get_neighborhood(self, size=5) -> np.ndarray:
        neighborhood = []

        for i in range(-(size // 2), (size // 2) + 1):
            row = []
            for j in range(-(size // 2), (size // 2) + 1):
                row.append(ContourPoint(self.x + i, self.y + j))
            neighborhood.append(row)

        result = np.empty((size, size), dtype=ContourPoint)
        result[:] = neighborhood

        return result

    def calculate_distance(self, from_point: ContourPoint) -> float:
        return np.sqrt(
            np.square(self.x - from_point.x) + np.square(self.y - from_point.y)
        )


# Represents a contour as a circular doubly linked list
class Contour:
    def __init__(self):
        self.start = None
        self.end = None
        self.num_points = 0

    def add_point(self, x: float, y: float):
        if self.num_points == 0:
            self.start = ContourPoint(x, y)
            self.end = self.start
            self.num_points = 1
            return

        new_point = ContourPoint(x, y)
        new_point.previous = self.end
        new_point.next = self.start
        self.end.next = new_point
        self.end = new_point
        self.start.previous = self.end
        self.num_points += 1

    def calculate_average_distance(self) -> float:
        distance = 0

        current = self.start.next

        for i in range(1, self.num_points):
            distance += current.calculate_distance(current.previous)
            current = current.next

        return distance / self.num_points

    def get_contour_array(self) -> np.ndarray:
        arr = np.empty(self.num_points, dtype=ContourPoint)
        current = self.start
        for i in range(self.num_points):
            arr[i] = (current.x, current.y)

        return arr

    def __len__(self):
        return self.num_points
