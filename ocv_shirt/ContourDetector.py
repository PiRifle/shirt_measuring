import cv2
import numpy as np
import numpy.typing as npt
import imutils
from ocv_shirt.stubs import Contours, BoundingBox

class ContourDetector:
    eps: float
    threshold: int
    maxval: int

    def __init__(self, eps:float = 0.005, threshold:int = 140, maxval:int = 255):
        self.eps, self.threshold, self.maxval = eps, threshold, maxval
        
    def analyze_image(self, frame: "cv2.Mat"):
        thresh = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.threshold, self.maxval, cv2.THRESH_BINARY_INV)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts_extracted = imutils.grab_contours(cnts)
        c = max(cnts_extracted, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx: Contours = cv2.approxPolyDP(c, self.eps * peri, True)
        box: BoundingBox = cv2.boundingRect(approx)        
        return approx, box, thresh
