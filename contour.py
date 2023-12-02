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

    def get_neighborhood(self, max_height: int, max_width: int, size=3) -> np.ndarray:
        neighborhood = np.empty((size, size), dtype=ContourPoint)

        row_ind = 0
        col_ind = 0
        for i in range(-(size // 2), (size // 2) + 1):
            for j in range(-(size // 2), (size // 2) + 1):
                x = max(0, min(self.x + i, max_height - 1))
                y = max(0, min(self.y + j, max_width - 1))
                neighborhood[row_ind][col_ind] = ContourPoint(
                    x, y, next=self.next, previous=self.previous
                )
                col_ind += 1
            row_ind += 1
            col_ind = 0

        return neighborhood

    @staticmethod
    def calculate_distance(point_1, point_2) -> float:
        return np.sqrt(
            np.square(point_2.x - point_1.x) + np.square(point_2.y - point_1.y)
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
            self.end.previous = self.start
            self.end.next = self.start
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
            distance += ContourPoint.calculate_distance(current.previous, current)
            current = current.next

        return distance / self.num_points

    def get_contour_array(self) -> np.ndarray:
        arr = np.empty(self.num_points, dtype=ContourPoint)
        current = self.start
        for i in range(self.num_points):
            arr[i] = (current.x, current.y)
            current = current.next

        return arr

    def get_contour_polyline(self) -> np.ndarray:
        arr = np.empty((self.num_points, 2), dtype=np.int32)

        current = self.start
        for i in range(self.num_points):
            arr[i] = [current.x, current.y]
            current = current.next

        return arr.reshape((-1, 1, 2))

    def __len__(self):
        return self.num_points

    @staticmethod
    def build_contour_from_array(arr: np.ndarray):
        contour = Contour()

        for i in range(arr.size):
            contour.add_point(arr[i][0], arr[i][1])

        return contour
