from ip import IP
import cv2
import numpy as np

class Resize(IP):
    def __init__(self, path):
        super().__init__(path)

    def resize(self, w, h):
        return cv2.resize(self.img, (w, h))
    
    def resize_nn(self,width,height):
        return cv2.resize(self.img, (width,height), interpolation=cv2.INTER_NEAREST)

    def resize_bilinear(self,width,height):
        return cv2.resize(self.img, (width,height), interpolation=cv2.INTER_LINEAR)

    def resize_bicubic(self,width,height):
        return cv2.resize(self.img, (width,height), interpolation=cv2.INTER_CUBIC) 
    