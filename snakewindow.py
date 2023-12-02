import numpy as np
import cv2 as cv

from contour import Contour, ContourPoint
from image import Image
from snake import Snake

IMAGE_PATH = "test.jpg"


class SnakeWindow(object):
    def __init__(self, image_path=IMAGE_PATH):
        self.WINDOW_NAME = "Snake Demo"
        self.window = cv.namedWindow(self.WINDOW_NAME)
        self.contour = Contour()
        self.image = Image(image_path)
        self.display_image = self.image.image_data.copy()
        self.display_image = cv.cvtColor(self.display_image, cv.COLOR_GRAY2BGR)
        cv.imshow(self.WINDOW_NAME, self.display_image)
        cv.setMouseCallback(self.WINDOW_NAME, self.draw_contour)

        cv.waitKey(0)
        cv.destroyAllWindows()

    def draw_contour(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.contour.add_point(x, y)
            cv.circle(
                self.display_image, (x, y), radius=5, color=(0, 255, 0), thickness=1
            )
            cv.imshow(self.WINDOW_NAME, self.display_image)
        elif event == cv.EVENT_RBUTTONDOWN:
            cv.polylines(
                self.display_image,
                [self.contour.get_contour_polyline()],
                True,
                (0, 0, 255),
                1,
                cv.LINE_AA,
            )
            cv.imshow(self.WINDOW_NAME, self.display_image)
        elif event == cv.EVENT_MBUTTONDOWN:
            self.start_snake()

    def update_contour(self, contour: Contour):
        self.contour = Contour.build_contour_from_array(contour.get_contour_array())
        self.display_image = cv.cvtColor(self.image.image_data, cv.COLOR_GRAY2BGR)

        for point in self.contour.get_contour_array():
            cv.circle(
                self.display_image,
                (point[0], point[1]),
                radius=5,
                color=(0, 255, 0),
                thickness=1,
            )

        cv.polylines(
            self.display_image,
            [self.contour.get_contour_polyline()],
            True,
            (0, 0, 255),
            1,
            cv.LINE_AA,
        )
        cv.imshow(self.WINDOW_NAME, self.display_image)

    def start_snake(self):
        snake = Snake(
            self.contour.get_contour_array(),
            IMAGE_PATH,
            alpha=1,
            beta=1,
            gamma=1.2,
        )

        for i in range(100):
            points_changed = snake.run_algorithm()
            self.update_contour(snake.contour)
            cv.waitKey(1)

        # while True:
        #     points_changed = snake.run_algorithm()
        #     self.update_contour(snake.contour)
        #     cv.waitKey(1)
        #     if points_changed <= 5:
        #         break
