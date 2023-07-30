import cv2
import numpy as np
import numpy.typing as npt

class Clothing:
    img: "cv2.Mat"
    edges: npt.NDArray
    thresh: npt.NDArray
    
    def render(self):
        img = self.img.copy()
        cv2.drawContours(img, [self.edges], -1, (0, 255, 0), 3)
        cv2.imshow("thresh", self.thresh)
        cv2.imshow("edges", img)
        cv2.imshow("img", self.img)
    
    @property
    def type(self):
        pass
    
    @property
    def color(self):
        pass
    
    def measure(self):
        pass
    
    def __init__(self, img: npt.NDArray, thresh:npt.NDArray, edges: npt.NDArray):
        self.img = img
        self.thresh = thresh
        self.edges = edges
    
    