import numpy as np
import cv2
from ocv_shirt.Camera import Camera2CV
import logging
from threading import Thread
from skimage.metrics import structural_similarity as compare_ssim

class CaptureEngine:
    last_scores: list[float] = [1]
    buff_len: int
    base: "cv2.Mat" | None = None
    cam: Camera2CV
    threshold: float
    worker_thread: Thread
    thread_running: bool = False
    
    def __init__(self, cam: Camera2CV, threshold: float = 0.88, buff_len: int = 20):
        if buff_len < 11:
            raise ValueError("buff_len too short")
        self.cam = cam
        self.threshold = threshold
        self.buff_len = buff_len
        self.worker_thread = Thread(target=self.__worker_loop)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self):
        self.stop()

    def append_to_last(self, val):
        val = round(val,1)
        self.last_scores.append(val)
        self.last_scores = self.last_scores[-self.buff_len:]

    def start(self):
        self.thread_running = True
        self.worker_thread.start()

    def stop(self):
        self.thread_running = False

    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def got_impulse(self):
        minimum = min(self.last_scores)
        maximum = max(self.last_scores)
        if len(self.last_scores) > 2:
            last_ten = self.last_scores[-10:-1]
        else:
            return False
        avg = round(sum(last_ten)/len(last_ten), 1)
        is_not_changing = avg == self.last_scores[-1]
        print(minimum, maximum, avg)
        return (minimum != maximum and is_not_changing and minimum != self.last_scores[-1] and maximum != self.last_scores[-1]) or maximum < self.threshold
        
    def __worker_loop(self):
        prev = None
        with self.cam as camera: # type: ignore # idk what the hell is up with that
            while self.thread_running:
                if self.got_impulse() and len(self.last_scores) > self.buff_len-1:
                    self.last_scores = [1]
                    self.base = prev
                    self.__capture(camera.capture())

                # sometimes the preview locks up, dont crash thread, leave last frame as preview
                try:
                    prev = camera.get_preview()
                except:
                    pass  
                if prev:
                    if self.base is None:
                        self.base = prev
                    (score, diff) = compare_ssim(cv2.cvtColor(self.base, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), full=True)
                    diff = (diff * 255).astype("uint8")
                    self.append_to_last(score)
                    self.__preview(prev, diff, score)
   
    
    def __capture(self, frame: "cv2.Mat"):
        self.on_capture_frame(frame)
    
    def __preview(self, frame: "cv2.Mat", difference: "cv2.Mat", score):
        self.on_preview_frame(frame)
        # thresh = cv2.threshold(difference, 100, 255, cv2.THRESH_BINARY_INV)

    
    def on_preview_frame(self, frame: "cv2.Mat"):
        pass
    
    def on_capture_frame(self, frame: "cv2.Mat"):
        pass
    
    