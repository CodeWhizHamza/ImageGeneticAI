# plotting color differences using plt and color library

import cv2
import numpy as np
import matplotlib.pyplot as plt
import colour

# Load target image
image_name = "eren"
target_image_path = f"source_images/cbpunk.jpg"

target_image = cv2.imread(target_image_path)
height, width, _ = target_image.shape

TARGET_WIDTH = 400

if width > TARGET_WIDTH:
    target_image = cv2.resize(
        target_image, (TARGET_WIDTH, int(height * TARGET_WIDTH / width))
    )

width, height, _ = target_image.shape

# Load canvas
canvas = cv2.imread("source_images/image.png")
canvas = cv2.resize(canvas, (height, width))

# Calculate color differences
differences = colour.difference.delta_E_CIE1976(canvas, target_image)

# Plot the differences
plt.imshow(differences, cmap="magma_r")
plt.colorbar()
plt.show()


