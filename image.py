import cv2 as cv
import numpy as np


class Image:
    def __init__(self, image_path):
        self.image = cv.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

    def show(self):
        cv.imshow("image", self.image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def get_gradient_magnitude(self) -> np.ndarray:
        gradient_x = cv.Sobel(self.image, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
        gradient_y = cv.Sobel(self.image, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
        return np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    def blur_image(self):
        return cv.GaussianBlur(self.image, (5, 5), 0)

    def threshold_image(image):
        return cv.adaptiveThreshold(
            iamge,
            maxValue=255,
            adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv.THRESH_BINARY,
        )
