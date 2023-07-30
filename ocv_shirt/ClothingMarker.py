from ocv_shirt import models
import cv2
from ocv_shirt.recognition import ClothingRecognizer
from ocv_shirt.ContourDetector import ContourDetector

class ClothingMarker:
    def __init__(self, eps = 0.005, threshold = 140, maxval = 255, model_path = None):
        if model_path:
            self.recognition = ClothingRecognizer(eps, threshold, maxval, model_path)
        else:
            self.recognition = ClothingRecognizer(eps, threshold, maxval)
            
    def analyze_image(self, frame:"cv2.Mat"):
        cnts, thresh, ps = self.recognition.analyze_export_contours(frame)
        idx = ps.index(max(ps))
        ps_mapping = {0: models.Shirt,
                                      1: models.Pants}
        return ps_mapping.get(idx, models.Clothing)(frame, thresh, cnts)