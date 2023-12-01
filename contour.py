import numpy as np


class ContourPoint:
    def __init__(self, x: float, y: float, next=None, previous=None):
        self.x = x
        self.y = y
        self.next = next
        self.previous = previous

    def update_point(self, new_x: float, new_y: float):
        self.x = new_x
        self.y = new_y

    def get_neighborhood(self, size=5) -> np.ndarray:
        neighborhood = np.empty((size, size), dtype=ContourPoint)

        for i in range(size):
            for j in range(size):
                neighborhood[i][j] = ContourPoint(
                    self.x - ((size // 2) + i), self.y - ((size // 2) + j)
                )

        return neighborhood

    def calculate_distance(self, from_point: ContourPoint) -> float:
        return np.sqrt(
            np.square(self.x - from_point.x) + np.square(self.y - from_point.y)
        )

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


# Represents a contour as a circular doubly linked list
class Contour:
    def __init__(self):
        self.start = None
        self.end = None
        self.num_points = 0

    def add_point(self, x: float, y: float):
        new_point = ContourPoint(x, y)

        if self.num_points == 0:
            self.start = ContourPoint(x, y)
            self.end = self.start
            self.start.next = self.end
            self.start.previous = self.end
            self.num_points = 1
            return

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
