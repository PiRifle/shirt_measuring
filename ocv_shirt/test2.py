import cv2

from MarkerDetector import MarkerDetector
marker_detect = MarkerDetector()
from ClothingMarker import ClothingMarker

import pulp
img = cv2.imread("../images/a.jpg")

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.namedWindow("f", cv2.WINDOW_NORMAL)
cv2.namedWindow("draw", cv2.WINDOW_NORMAL)

cv2.imshow('img', img)

markers = {0:1, 1:2, 2:2, 3:1}
markers_positions = {2:1, 3:2, 0:3, 1:0}

marker_detect.set_points(markers, markers_positions)

marker_detect.auto_adjust_params(img, [0,1,2,3])

from PerspectiveCorrector import PerspectiveCorrector

perspective_corrector = PerspectiveCorrector((900, 1500), 1)

box = marker_detect.analyze_image(img)

corrected =  perspective_corrector.process_frame(img, box)

cv2.imshow("img", img)

cv2.imshow("f", corrected)

cont = corrected.copy()

clothing_marker = ClothingMarker()

clothing = clothing_marker.analyze_image(corrected)

clothing.render()

cv2.waitKey(0)




