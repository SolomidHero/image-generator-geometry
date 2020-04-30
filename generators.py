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
    self._mask_color = {
      "background": 0,
      "rectangle": 1,
      "square": 1,
      "triangle": 2,
      "circle": 3,
    }


  def randomPoint(self, pad=10):
    return np.array([
      randint(pad, self._height - pad),
      randint(pad, self._width - pad)
    ])

  def addNoise(self, std=15):
    self.image += sps.norm(scale=std).rvs(self.image.shape).astype(np.uint8)
    self.image = self.image.clip(0, 255)

    return self

  def drawRectangle(self, pad=10):
    color = tuple(randint(0, 255, dtype=int) for _ in range(self._channels))
    corners = np.array([
      randint(pad, self._height - pad, 2),
      randint(pad, self._width - pad, 2)
    ])
    cv2.rectangle(self.image, tuple(corners[:, 0]), tuple(corners[:, 1]), color, -1)
    cv2.rectangle(self.mask, tuple(corners[:, 0]), tuple(corners[:, 1]), self._mask_color["rectangle"], -1)

    return self

  def drawSquare(self, pad=10):
    color = tuple(randint(0, 255, dtype=int) for _ in range(self._channels))
    center = self.randomPoint()
    max_size = randint(pad // 2, min([
      center[0],
      center[1],
      self._height - center[0],
      self._width - center[1]
    ]))
    cv2.rectangle(self.image, tuple(center - max_size), tuple(center + max_size), color, -1)
    cv2.rectangle(self.mask, tuple(center - max_size), tuple(center + max_size), self._mask_color["square"], -1)

    return self

  def drawTriangle(self, pad=10):
    color = tuple(randint(0, 255, dtype=int) for _ in range(self._channels))
    fig = np.array([
      randint(pad, self._height - pad, 3),
      randint(pad, self._width - pad, 3)
    ], dtype=int).T
    cv2.fillPoly(self.image, [fig], color)
    cv2.fillPoly(self.mask, [fig], self._mask_color["triangle"])

    return self

  def drawCircle(self, pad=10):
    color = tuple(randint(0, 255, dtype=int) for _ in range(self._channels))
    center = self.randomPoint()
    radius = randint(pad // 2, min([
      center[0],
      center[1],
      self._height - center[0],
      self._width - center[1]
    ]))
    cv2.circle(self.image, tuple(center), radius, color, -1)
    cv2.circle(self.mask, tuple(center), radius, self._mask_color["circle"], -1)

    return self

  def generateImage(self, figs="square"):
    if isinstance(figs, str):
      figs = [figs]
    for fig in figs:
      if fig not in self._draw_method.keys():
        raise ValueError(f"fig should be one of this: {self._draw_method.keys():}")

    self.mask = np.zeros((self._height, self._width), dtype=np.uint8)
    self.image = np.empty((self._height, self._width, 0), dtype=np.uint8)
    for _ in range(self._channels):
      self.image = np.append(
        self.image,
        np.full((self._height, self._width, 1), np.uint8(randint(0, 255))),
        axis=2
      )

    for fig in figs:
      self._draw_method[fig]()

    return self

  def get(self, masked=True):
    if masked:
      return self.image, self.mask

    return self.image
