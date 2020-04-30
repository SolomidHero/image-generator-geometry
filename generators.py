import numpy as np
from numpy.random import randint
import scipy.stats as sps
import cv2


class Drawer:
  """
  Generator of geometric shapes
  Args:
  - height, weight - shape of image
  - channels - number of channels in image
  """
  def __init__(self, height=128, width=128, channels=3):
    self._height = height
    self._width = width
    self._channels = channels
    self._draw_method = {
      "rectangle": self.drawRectangle,
      "square": self.drawSquare,
      "triangle": self.drawTriangle,
      "circle": self.drawCircle,
    }


  def addNoise(self, std=20):
    self.image += sps.norm(scale=std).rvs(self.image.shape).astype(np.uint8)
    self.image = self.image.clip(0, 255)


  def drawRectangle(self, pad=10):
    color = tuple(randint(0, 255, dtype=int) for _ in range(self._channels))
    corners = np.array([
      randint(pad, self._height - pad, 2),
      randint(pad, self._width - pad, 2)
    ])
    cv2.rectangle(self.image, tuple(corners[:, 0]), tuple(corners[:, 1]), color, -1)


  def drawSquare(self, pad=10):
    color = tuple(randint(0, 255, dtype=int) for _ in range(self._channels))
    center = randint(pad, self._height - pad, 2)
    max_size = randint(0, min([
      center[0] - pad,
      center[1] - pad,
      self._height - pad - center[0],
      self._width - pad - center[1]
    ])) + pad
    cv2.rectangle(self.image, tuple(center - max_size), tuple(center + max_size), color, -1)


  def drawTriangle(self, pad=10):
    color = tuple(randint(0, 255, dtype=int) for _ in range(self._channels))
    fig = np.array([
      randint(pad, self._height - pad, 3),
      randint(pad, self._width - pad, 3)
    ], dtype=int).T
    cv2.fillPoly(self.image, [fig], color)


  def drawCircle(self, pad=10):
    color = tuple(randint(0, 255, dtype=int) for _ in range(self._channels))
    center = randint(pad, self._height - pad, 2)
    radius = randint(0, min([
      center[0] - pad,
      center[1] - pad,
      self._height - pad - center[0],
      self._width - pad - center[1]
    ])) + pad
    cv2.circle(self.image, tuple(center), radius, color, -1)


  def GenerateImage(self, fig="square"):
    self.image = np.empty((self._height, self._width, 0), dtype=np.uint8)
    for _ in range(self._channels):
      self.image = np.append(
        self.image,
        np.full((self._height, self._width, 1), np.uint8(randint(0, 255))),
        axis=2
      )

    self._draw_method[fig]()

    self.addNoise()

    return self.image

