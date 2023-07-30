import cv2
import numpy as np
import imutils

from typing import cast

class PerspectiveCorrector:
    dpcm: int
    dimensions: tuple[int, int]
    
    def __init__(self, dimensions: tuple[float, float], dpcm: int):
        self.dpcm = dpcm
        
        w, h = dimensions
        self.dimensions = int(w * dpcm), int(h * dpcm)
        
    def _stretch_image(self, image, w, h):
        # Get the current dimensions of the image
        height, width = image.shape[:2]

        # Apply the scaling factors to resize the image
        resized_image = cv2.resize(image, (w, h))

        return resized_image
    
    def process_frame(self, frame: "cv2.Mat", points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]) -> "cv2.Mat":
        tl, tr, br, bl = points
        
        # Finding the maximum width.
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # Finding the maximum height.
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # Final destination co-ordinates.
        destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
        
        M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), np.float32(destination_corners)) #type: ignore

        warped_image = cv2.warpPerspective(frame, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LINEAR)
        aspect_corrected = self._stretch_image(warped_image, *self.dimensions)
        
        #cast to "cv2.Mat" for peace of mind
        return cast("cv2.Mat", aspect_corrected)