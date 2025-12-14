from ip import IP
import cv2
import numpy as np


class Transformation(IP):
    def __init__(self, path):
        super().__init__(path)
    
    def crop(self,x,y,w,h):
        return self.img[y:y+h, x:x+w]

    def rotate(self, angle):
        M = cv2.getRotationMatrix2D((self.width / 2, self.height / 2), angle, 1)
        return cv2.warpAffine(self.img, M, (self.width, self.height))

    def shear_x(self, factor=0.3):
        M = np.array([[1, factor, 0],[0, 1, 0]], dtype=np.float32)
        sheared = cv2.warpAffine(self.img, M, (self.width + int(self.height * abs(factor)), self.height))
        return sheared

    def shear_y(self, factor=0.3):
        M = np.array([[1, 0, 0],[factor, 1, 0]], dtype=np.float32)
        sheared = cv2.warpAffine(self.img, M, (self.width, self.height + int(self.width * abs(factor))))
        return sheared

    def translate(self,tx=0,ty=0):
        M=np.array([[1,0,tx],[0,1,ty]], dtype=np.float32)
        translated=cv2.warpAffine(self.img,M,(self.width,self.height))
        return translated