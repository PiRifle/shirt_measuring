import torch
from ocv_shirt.ai.CIFAR10Model import CIFAR10Model
from ocv_shirt.ContourDetector import ContourDetector
import numpy as np
import numpy.typing as npt
import cv2
import os

class ClothingRecognizer:
    model = CIFAR10Model()
    contour_reg: ContourDetector
    labels = ['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot']
    
    def __init__(self, eps=0.005, threshold=140, maxval=255, model_path = os.path.join(os.path.dirname(__file__), "ai/ai.pth")):
        self.contour_reg = ContourDetector(eps, threshold, maxval)
        self.model.eval()
        self.model.load_state_dict(torch.load(model_path))

    def __crop_and_center_image(self, image: "cv2.Mat", bounding_box: tuple[int, int, int, int]):
        x, y, width, height = bounding_box
        target_width = max(width, height)
        target_height = int(target_width)
        crop_x = x
        crop_y = y
        crop_x2 = crop_x + width
        crop_y2 = crop_y + height
        cropped_image = image[crop_y:crop_y2, crop_x:crop_x2]
        paste_x = max(-(crop_x - x), 0)
        paste_y = max(-(crop_y - y), 0)
        canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
        center_x = paste_x + cropped_image.shape[1] // 2
        center_y = paste_y + cropped_image.shape[0] // 2
        canvas_center = np.array([canvas.shape[1] // 2, canvas.shape[0] // 2])
        paste_position = canvas_center - np.array([center_x, center_y]) + np.array([paste_x, paste_y])
        canvas[paste_position[1]:paste_position[1] + cropped_image.shape[0],
            paste_position[0]:paste_position[0] + cropped_image.shape[1]] = cropped_image

        return canvas

    def __scale_range(self, arr: npt.NDArray) -> npt.NDArray:
        min_val = np.min(arr)
        max_val = np.max(arr)
        return 2 * (arr - min_val) / (max_val - min_val) - 1


    
    def label_image(self, frame: "cv2.Mat") -> str:
        ps = self.analyze_image(frame)
        return self.labels[ps.index(max(ps))]
        
    def analyze_image(self, frame: "cv2.Mat") -> list[float]:
        frame = frame.copy()
        _, box, thresh = self.contour_reg.analyze_image(frame)
        frame[thresh == 0] = (255, 255, 255)
        out = cv2.cvtColor(cv2.resize(self.__crop_and_center_image(frame, box), (28, 28)), cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(out).reshape(1, 1, 28, 28).astype('float32') / 255
        img = torch.tensor(self.__scale_range(inverted))
        ps = torch.exp(self.model(img)).tolist()
        return ps
    
    def analyze_export_contours(self, frame: "cv2.Mat"):
        frame = frame.copy()
        approx_cont, box, thresh = self.contour_reg.analyze_image(frame)
        frame[thresh == 0] = (255, 255, 255)
        out = cv2.cvtColor(cv2.resize(self.__crop_and_center_image(frame, box), (28, 28)), cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(out).reshape(1, 1, 28, 28).astype('float32') / 255
        img = torch.tensor(self.__scale_range(inverted))
        ps = torch.exp(self.model(img)).tolist()
        return approx_cont, thresh, ps