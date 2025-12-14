import cv2
import numpy as np
from ip import IP

class Filter(IP):
    def __init__(self, path):
        super().__init__(path)

    def to_gray(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def to_blur(self, k=19,sigma=3):
        return cv2.GaussianBlur(self.img, (k, k), 3)

    def median(self,k=7):
        return cv2.medianBlur(self.img,k)

    def laplacian(self,k=3):
        gray= self.to_gray()
        lap= cv2.convertScaleAbs(cv2.Laplacian(gray,cv2.CV_64F,ksize=k))
        return lap

    def sobel(self,k=3):
        gray= self.to_gray()
        soblex=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=k)
        sobley=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=k)
        return cv2.convertScaleAbs(cv2.magnitude(soblex,sobley))

    def gradient(self):
        gray=self.to_gray()
        gx = cv2.filter2D(gray,cv2.CV_64F, np.array([[-1,0,1],[-1,0,1],[-1,0,1]]))
        gy = cv2.filter2D(gray,cv2.CV_64F, np.array([[-1,-1,1],[0,0,0],[1,1,1]]))
        return cv2.convertScaleAbs(cv2.magnitude(gx,gy))

    def sharpen(self):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(self.img, -1, kernel)

    def to_binary(self):
        gray = self.to_gray()
        avg = gray.mean()
        _, result = cv2.threshold(gray, avg, 255, cv2.THRESH_BINARY)
        return result, avg

    def adjust(self, brightness=0, contrast=0):
        return cv2.convertScaleAbs(self.img, alpha=1 + contrast / 100, beta=brightness)
