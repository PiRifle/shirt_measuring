import gphoto2cffi as gp
from gphoto2cffi.backend import lib
import numpy as np
import cv2

class Camera2CV(gp.Camera):
    def __init__(self, host=None):
        if host:
            super().ptpip(host)
        else:
            super().__init__()
        
    def __enter__(self):
        return self
    
    def capture(self) -> "cv2.Mat":
        target = self.config['settings']['capturetarget']
        target.set("Internal RAM")   
        lib.gp_camera_trigger_capture(self._cam, self._ctx)
        fobj = self._wait_for_event(event_type=lib.GP_EVENT_FILE_ADDED)
        data = fobj.get_data()
        try:
            fobj.remove()
        except:
            pass
        img = np.frombuffer(data, np.uint8)
        img_dec = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)
        self._wait_for_event(lib.GP_EVENT_CAPTURE_COMPLETE)

        return img_dec
        
    def get_preview(self) -> "cv2.Mat":
        data = super().get_preview()
        img = np.frombuffer(data, np.uint8)
        img_dec = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)
        
        return img_dec

    def __exit__(self):
        self.exit()