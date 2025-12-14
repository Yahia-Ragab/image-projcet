from ip import IP
import os

class Info(IP):

    def __init__(self, path):
        super().__init__(path)

    def get_resolution(self):
        return self.width, self.height

    def get_size(self):
        size = os.path.getsize(self.path) / (1024 * 1024)
        return round(size, 1)

    def get_type(self):
        return os.path.splitext(self.path)[1].upper().replace('.', '')

    def get_channel(self):
        return self.img.shape[2] if len(self.img.shape) == 3 else 1
