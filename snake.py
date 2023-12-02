from scipy.signal import argrelextrema
from contour import *
from image import *


class Snake:
    def __init__(
        self,
        contour_array: np.ndarray,
        image_path: str,
        alpha: float = 1,
        beta: float = 1,
        gamma: float = 1.2,
    ):
        self.contour = Contour.build_contour_from_array(contour_array)
        self.curvature_threshold = 0.01
        self.alpha = np.full(self.contour.num_points, alpha)
        self.beta = np.full(self.contour.num_points, beta)
        self.gamma = np.full(self.contour.num_points, gamma)
        self.image = Image(image_path)

    def continuity_energy_at_point(self, point: ContourPoint) -> float:
        return np.square(
            self.contour.calculate_average_distance()
            - ContourPoint.calculate_distance(point.previous, point)
        )

    def continuity_energy_for_neighborhood(
        self, neighborhood: np.ndarray
    ) -> np.ndarray:
        energy = np.zeros((neighborhood.shape[0], neighborhood.shape[1]))

        for i in range(neighborhood.shape[0]):
            for j in range(neighborhood.shape[1]):
                energy[i][j] = self.continuity_energy_at_point(neighborhood[i][j])

        return self.normalize_energy(energy, np.max(energy), np.min(energy))

    def curvature_energy_at_point(self, point: ContourPoint) -> float:
        return np.square(point.previous.x - (2 * point.x) + (point.next.x)) + np.square(
            point.previous.y - (2 * point.y) + (point.next.y)
        )

    def curvature_energy_for_neighborhood(self, neighborhood: np.ndarray) -> np.ndarray:
        energy = np.zeros((neighborhood.shape[0], neighborhood.shape[1]))

        for i in range(neighborhood.shape[0]):
            for j in range(neighborhood.shape[1]):
                energy[i][j] = self.curvature_energy_at_point(neighborhood[i][j])

        return self.normalize_energy(energy, np.max(energy), np.min(energy))

    def image_energy_at_point(self, point: ContourPoint) -> float:
        return -np.float64(self.image.gradient_magnitude[point.y][point.x])

    def image_energy_for_neighborhood(self, neighborhood: np.ndarray) -> np.ndarray:
        energy = np.zeros((neighborhood.shape[0], neighborhood.shape[1]))

        for i in range(neighborhood.shape[0]):
            for j in range(neighborhood.shape[1]):
                energy[i][j] = self.image_energy_at_point(neighborhood[i][j])

        return self.normalize_energy(energy, np.max(energy), np.min(energy))

    def normalize_energy(self, energy, max_energy, min_energy):
        if max_energy == min_energy:
            return np.zeros(energy.shape)

        return (energy - min_energy) / (max_energy - min_energy)

    def total_energy_at_index(
        self,
        point_index: int,
        e_cont: np.ndarray,
        e_curv: np.ndarray,
        e_img: np.ndarray,
    ):
        return (
            (self.alpha[point_index] * e_cont)
            + (self.beta[point_index] * e_curv)
            + (self.gamma[point_index] * e_img)
        )

    def calculate_curvature_for_contour(self) -> np.ndarray:
        k = np.zeros(self.contour.num_points)

        current = self.contour.start
        for i in range(self.contour.num_points):
            k[i] = self.curvature_energy_at_point(current)
            current = current.next

        return self.normalize_energy(k, np.max(k), np.min(k))

    def run_algorithm(self) -> int:
        num_points_changed = 0
        current = self.contour.start

        # Step 1
        for i in range(self.contour.num_points):
            neighborhood = current.get_neighborhood(
                max_height=self.image.width, max_width=self.image.height
            )
            e_cont = self.continuity_energy_for_neighborhood(neighborhood)
            e_curv = self.curvature_energy_for_neighborhood(neighborhood)
            e_img = self.image_energy_for_neighborhood(neighborhood)
            total = self.total_energy_at_index(i, e_cont, e_curv, e_img)

            min_energy = np.argmin(total)
            min_index = np.unravel_index(min_energy, total.shape)

            min_point = neighborhood[min_index[0]][min_index[1]]
            if min_point != current:
                current.update_point(min_point.x, min_point.y)
                num_points_changed += 1
            current = current.next

        # Step 2
        k = self.calculate_curvature_for_contour()
        local_maxima = argrelextrema(k, np.greater)[0]
        for i in range(len(local_maxima)):
            if k[local_maxima[i]] > self.curvature_threshold:
                self.beta[local_maxima[i]] = 0

        return num_points_changed
