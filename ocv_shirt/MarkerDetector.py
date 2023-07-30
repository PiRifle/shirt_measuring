import cv2
import numpy as np
import numpy.typing as npt

from cv2 import aruco

from ocv_shirt.lib.matrix import spiralMatrix

class MarkerDetector:
    aruco_dict: aruco.Dictionary
    aruco_params: aruco.DetectorParameters
    contrast: int = 255
    brightness: int = 127
    markers: dict[int, int] = {}
    marker_mapping: dict[int, int] = {}
    last_bc_config = (254//4, 127//3)
    
    def set_contrast(self, contrast):
        self.contrast = contrast
        
    def set_brightness(self, brightness):
        self.brightness = brightness

    def _adjust_contrast_brightness(self, frame: "cv2.Mat", brightness: int = 255, contrast: int = 127) -> "cv2.Mat":
        brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
        contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                max = 255
            else:
                shadow = 0
                max = 255 + brightness
            al_pha = (max - shadow) / 255
            ga_mma = shadow
            cal = cv2.addWeighted(frame, al_pha, 
                                frame, 0, ga_mma)
        else:
            cal = frame
        if contrast != 0:
            Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            Gamma = 127 * (1 - Alpha)
            cal = cv2.addWeighted(cal, Alpha, 
                                cal, 0, Gamma)    
        return cal
    
    def __init__(self, marker_db: int = aruco.DICT_4X4_50 ):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params =  cv2.aruco.DetectorParameters()
    
    def auto_adjust_params(self, frame, marker_ids: list[int], iterations = 1000):
        spiral = spiralMatrix(254//2, 127//2, *self.last_bc_config)
        for _ in range(iterations):
            y, x = next(spiral)
            c, b = x * 4, y * 4
            self.set_brightness(b)
            self.set_contrast(c)
            corners, ids, _corr, _ = self.analyze_image(frame, debug=True)
            # cv2.imshow("img", corr)
            # cv2.waitKey(1)
            if ids is None:
                ids = []
            else:
                ids = ids.flatten()
            if set(list(ids)) == set(marker_ids):
                self.last_bc_config = (b, c)
                break

    def set_points(self, markers: dict[int, int], marker_mapping: dict[int, int] | None = None):
        """applies ids of markers to given image corner

        Args:
            markers (dict[int, int]): ID of marker -> edge of marker
            marker_mapping (dict[int, int]): ID of marker -> edge of plane
        """
        if len(markers.keys()) != 4:
            raise ValueError("Marker count is not equal 4")
        
        if marker_mapping is None:
            marker_mapping = {0:0, 1:1, 2:2, 3:3}
            
        if markers.keys() != marker_mapping.keys():
            raise ValueError("Marker mapping doesnt have the same keys as the marker dict")
            
        self.markers = markers
        self.marker_mapping = marker_mapping

    def analyze_image(self, frame: "cv2.Mat", return_if_incomplete=False, debug=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corrected_frame = self._adjust_contrast_brightness(gray, self.brightness, self.contrast)
        (corners, ids, rejected) = cv2.aruco.detectMarkers(corrected_frame, self.aruco_dict, parameters=self.aruco_params)
        
        if debug:
            return corners, ids, corrected_frame, gray
        
        # imgxy = image.shape[1], image.shape[0]
        if ids is not None and len(ids) > 0:
            ids = ids.flatten()
                
            markers: dict[int, npt.ArrayLike] = dict(zip(ids, [c.reshape(4,2) for c in corners]))
                
               
            if markers.keys() != self.markers.keys():
                if return_if_incomplete:
                    return markers
                return None
            
            def get_marker_edge(marker_id: int, coordinates: npt.ArrayLike) -> tuple[int, int]:
                return tuple(int(x) for x in coordinates[self.markers.get(marker_id, 0)])
            
            def get_screen_edge(marker_id: int):
                return self.marker_mapping.get(marker_id, None)
            
            _marker_selected = dict((_id, get_marker_edge(_id, coord)) for _id, coord in markers.items())
            
            screen_edge = dict((get_screen_edge(marker_id), _marker_selected[marker_id]) for marker_id in list(self.markers.keys()))
            
            sorted_items = sorted(screen_edge.items(), key=lambda item: item[0])

            return tuple(item[1] for item in sorted_items)