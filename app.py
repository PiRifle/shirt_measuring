import cv2
from utils import controller, stretch_image, crop_and_center_image
import numpy as np
import imutils

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()

vid = cv2.VideoCapture(0)

_contrast, _brightness = 0,0

def on_change_contrast(value):
    global _contrast
    _contrast = value
def on_change_brightness(value):
    global _brightness
    _brightness = value

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("aspect_corrected", cv2.WINDOW_NORMAL)

cv2.createTrackbar('contrast', "frame", 0, 255, on_change_contrast)
cv2.createTrackbar('brightness', "frame", 0, 512, on_change_brightness)

while(True):
    
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _aruco_out = controller(gray, contrast=_contrast, brightness=_brightness)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(_aruco_out, arucoDict, parameters=arucoParams)     
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if ids is None:
        cv2.imshow("frame", _aruco_out)
        continue
    
    ids = ids.flatten()
    imgxy = frame.shape[1], frame.shape[0]
    points = {}
    markers = dict(zip(ids, corners))
    for _id, coords in markers.items():
        corners = coords.reshape((4, 2))
        if _id in [0,3]:
            point = corners[1]
        else:
            point = corners[2]
        points[_id] = [int(i) for i in point]
    
    
    # point_bg = frame.copy()
    point_bg = _aruco_out.copy()
    for point in points.values():
        cv2.circle(point_bg, point, 16, (0,255,0), 4)
    cv2.imshow("frame", point_bg)
    
    if len(ids) != 4:
        continue
    
    tl = points[1]
    tr = points[2]
    bl = points[0]
    br = points[3]
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), np.float32(destination_corners))

    warped_image = cv2.warpPerspective(frame, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LINEAR)

    dest_w, dest_h = (900, 1500)
    aspect_corrected = stretch_image(warped_image, dest_w, dest_h)
    
    thresh = cv2.threshold(cv2.cvtColor(aspect_corrected, cv2.COLOR_BGR2GRAY), 140, 255, cv2.THRESH_BINARY_INV)[1]
    
    # cv2.imshow("aspect_corrected", aspect_corrected)

    # cv2.imshow("threshold", thresh)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    eps = 0.005
    
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps * peri, True)

    output = aspect_corrected.copy()
    show_box = cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
    box = cv2.boundingRect(approx)
    (x, y, w, h) = box

    show_box = cv2.rectangle(show_box, (x, y), (x + w, y + h), (255,0,0), 4)
    text = "original, num_pts={}".format(len(approx))
    show_box = cv2.putText(show_box, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (0, 255, 0), 2)
    
    cv2.imshow("show_box", np.concatenate([aspect_corrected, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), show_box], axis=1))

  
vid.release()
cv2.destroyAllWindows()