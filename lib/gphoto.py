import gphoto2cffi as gp
from gphoto2cffi.backend import lib
import logging
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import imutils
import matplotlib.pyplot as plt
import math
import time

def capture_image(cam: gp.Camera) -> "cv2.Mat":

    target = cam.config['settings']['capturetarget']
    target.set("Internal RAM")
    lib.gp_camera_trigger_capture(cam._cam, cam._ctx)
    fobj = cam._wait_for_event(event_type=lib.GP_EVENT_FILE_ADDED)

    data = fobj.get_data()

    try:
        fobj.remove()
    except:
        pass

    img = np.frombuffer(data, np.uint8)
    img_dec = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)
    cam._wait_for_event(lib.GP_EVENT_CAPTURE_COMPLETE)
    return img_dec

def retrieve_preview(cam: gp.Camera):
    img = np.frombuffer(my_cam.get_preview(), np.uint8)
    img_dec = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)
    return img_dec

logging.basicConfig(level=logging.DEBUG)

# Get an instance for the first supported camera
my_cam = gp.Camera()
# or
# my_cam = next(gp.list_cameras())

# print(my_cam.supported_operations)

# my_cam.

base = None

last_scores: list[float] = [1]


def append_to_last(val):
    global last_scores
    val = round(val,1)
    last_scores.append(val)
    last_scores = last_scores[-30:]


def got_impulse(arr):
    minimum = min(arr)
    maximum = max(arr)
    if len(arr) > 2:
        last_ten = arr[-10:-1]
    else:
        last_ten = [1]
    avg = round(sum(last_ten)/len(last_ten), 1)
    is_not_changing = avg == arr[-1]
    print(minimum, maximum, avg)
    return (minimum != maximum and is_not_changing and minimum != arr[-1] and maximum != arr[-1]) or maximum < 0.88
    
    
cv2.namedWindow("result", cv2.WINDOW_NORMAL) 
prev = None



while True:
    
    if got_impulse(last_scores) and len(last_scores) > 20:
        print("take picture")
        print(last_scores)
        last_scores = [1]
        base = prev
        # my_cam.exit()
        cv2.imshow("result", capture_image(my_cam))
        # if my_cam.status.liveviewprohibit:
            # fobj = my_cam._wait_for_event(event_type=lib.GP_EVENT_FILE_ADDED, duration=3000)
            # print(my_cam.status)
            # print(my_cam._cam)
            # break
    try:
        prev = retrieve_preview(my_cam)
    except:
        pass
    if base is None:
        base = prev
    (score, diff) = compare_ssim(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), full=True)
    diff = (diff * 255).astype("uint8")
    cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV, dst=diff)
    
    cv2.imshow("out", prev)   
    cv2.imshow("diff", diff)
    cv2.imshow("base", base)
    
    append_to_last(score)
    plt.cla()
    plt.plot(last_scores)
    plt.pause(0.05)
    
    
                
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.show()

my_cam.exit()


# img = np.frombuffer(my_cam.capture(), np.uint8)

# img_dec = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)

# cv2.imshow("out", img_dec)
# cv2.waitKey(0)

# Capture an image to the camera's RAM and get its data
# imgdata = my_cam.capture()

# Grab a preview from the camera
# previewdata = my_cam.get_preview()

# # Get a list of files on the camera
# files = tuple(my_cam.list_all_files())

# Iterate over a file's content
# with open("image.jpg", "wb") as fp:
#     for chunk in my_cam.files[0].iter_data():
#         fp.write(chunk)

# # Get a configuration value
# image_quality = my_cam.config['capturesettings']['imagequality'].value
# # Set a configuration value
# my_cam.config['capturesettings']['imagequality'].set("JPEG Fine")