import cv2
import numpy as np
from ip import IP
from filter import Filter

class Analysis(Filter):
    def __init__(self, path):
        super().__init__(path)

    def compute_threshold(self):
        binary_img, avg = self.to_binary()
        gray = self.to_gray()
        otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        difference = abs(avg - otsu_thresh)
        return {
            "average_threshold": avg,
            "otsu_threshold": otsu_thresh,
            "difference": difference,
            "is_optimal": difference < 10
        }

    def compute_histogram(self):
        gray=self.to_gray()
        hist=cv2.calcHist([gray],[0],None,[256],[0,256])
        return hist.flatten()
        