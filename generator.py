import numpy as np
import cv2
from random import randint


def GenerateImage(height=512, width=512, fig="square", channels=3):
  image = np.empty((height, width, 0), dtype=np.uint8)
  for i in range(channels):
    image = np.append(
      image,
      np.full((height, width, 1), np.uint8(randint(0, 255))),
      axis=2
    )
  print(image.shape)

  cv2.imshow(f"some_{fig}", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

GenerateImage()

