import cv2

class IP:

    def __init__(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Image not found or unreadable.")
        self.path = path
        self.img = img
        self.height, self.width = img.shape[:2]
