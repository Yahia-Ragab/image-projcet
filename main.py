from transformation import Transformation
from resize import Resize

# Image path (only once)
path = './assets/input.jpg'

# -------------------
# Transformation class
# -------------------
t = Transformation(path)

# Rotate 45 degrees
rotated_img = t.rotate(45)

# Shear in X direction
shear_x_img = t.shear_x(0.3)

# Shear in Y direction
shear_y_img = t.shear_y(0.3)

# Translate 50 pixels right and 30 pixels down
translated_img = t.translate(50, 30)

print("Transformed Images Shapes:")
print("Rotated:", rotated_img.shape)
print("Shear X:", shear_x_img.shape)
print("Shear Y:", shear_y_img.shape)
print("Translated:", translated_img.shape)

# -------------------
# Resize class
# -------------------
r = Resize(path)

# Default resize
resized_img = r.resize(200, 200)

# Nearest-neighbor
nearest_img = r.resize_nn(200, 200)

# Bilinear
bilinear_img = r.resize_bilinear(200, 200)

# Bicubic
bicubic_img = r.resize_bicubic(200, 200)

print("\nResized Images Shapes:")
print("Default:", resized_img.shape)
print("Nearest:", nearest_img.shape)
print("Bilinear:", bilinear_img.shape)
print("Bicubic:", bicubic_img.shape)
