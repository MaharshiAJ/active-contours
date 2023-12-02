import cv2 as cv
import numpy as np


class Image:
    def __init__(self, image_path):
        image = cv.imread(image_path)
        self.image_data = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.height, self.width = self.image_data.shape
        self.gradient_magnitude = None
        self.calculate_gradient_magnitude()

    def show(self):
        cv.imshow("image", self.image_data)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def calculate_gradient_magnitude(self):
        gradient_x = cv.Sobel(self.image_data, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
        gradient_y = cv.Sobel(self.image_data, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
        abs_gradient_x = cv.convertScaleAbs(gradient_x)
        abs_gradient_y = cv.convertScaleAbs(gradient_y)
        self.gradient_magnitude = cv.addWeighted(
            abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0
        )
        self.gradient_magnitude = Image.invert_image(self.gradient_magnitude)
        self.gradient_magnitude = Image.blur_image(self.gradient_magnitude)
        self.gradient_magnitude = Image.threshold_image(self.gradient_magnitude)

    def blur_image(image) -> np.ndarray:
        return cv.GaussianBlur(image, (3, 3), sigmaX=1, sigmaY=1)

    def threshold_image(image) -> np.ndarray:
        return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    def invert_image(image) -> np.ndarray:
        return cv.bitwise_not(image)
