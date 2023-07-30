import cv2
import numpy as np

def controller(img, brightness=255,
               contrast=127):
    
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
  
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha, 
                              img, 0, ga_mma)
  
    else:
        cal = img
  
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
  
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha, 
                              cal, 0, Gamma)
  
    # putText renders the specified text string in the image.
    cv2.putText(cal, 'B:{},C:{}'.format(brightness,
                                        contrast), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  
    return cal

def get_linear_max_1(p1, p2, max_coord): 
    x1, y1 = p1
    x2, y2 = p2
    xmax, ymax = max_coord
    
    # print("point1", p1, "point2", p2, "max_values", max_coord)
    
    a = (y1 - y2) / (x1 - x2)
    b = -(a*x1) + y1

    # print("a", a, "b", b)

    ox1 = (ymax-b) / a
    oy1 = a * xmax + b
    
    # print("ox1", ox1, "oy1", oy1)
    
    if ox1 >= xmax:
        return max(0,int(xmax)), max(0,int(oy1))
    else:
        return max(0,int(ox1)), max(0, int(ymax))
def get_linear_max_2(p1, p2, max_coord): 
    x1, y1 = p1
    x2, y2 = p2
    xmax, ymax = max_coord
    
    # print("point1", p1, "point2", p2, "max_values", max_coord)
    # 
    a = (y1 - y2) / (x1 - x2)
    b = -(a*x1) + y1

    # print("a", a, "b", b)

    ox1 = (ymax-b) / a
    oy1 = a * xmax + b
    
    # print("ox1", ox1, "oy1", oy1)
    
    if ox1 >= xmax:
        return max(0,int(xmax)), max(0,int(oy1))
    else:
        return max(0,int(ox1)), max(0, int(ymax))


# print(get_linear_max((4,1), (2,3), (20, 40)))

def stretch_image(image, new_width, new_height):
    # Get the current dimensions of the image
    height, width = image.shape[:2]

    # Calculate the scaling factors for width and height
    width_ratio = new_width / width
    height_ratio = new_height / height

    # Apply the scaling factors to resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


def crop_and_center_image(image, bounding_box):
    x, y, width, height = bounding_box
    # Calculate the aspect ratio of the bounding box
    aspect_ratio = 1

    # Determine the target dimensions for the crop
    target_width = max(width, height)
    target_height = int(target_width)

    # Calculate the top-left corner coordinates of the square crop
    crop_x = x
    crop_y = y

    # Calculate the bottom-right corner coordinates of the square crop
    crop_x2 = crop_x + width
    crop_y2 = crop_y + height

    # Adjust the crop coordinates to ensure they are within the image boundaries
    # crop_x = max(crop_x, 0)
    # crop_y = max(crop_y, 0)
    # crop_x2 = min(crop_x2, image.shape[1])
    # crop_y2 = min(crop_y2, image.shape[0])

    # Crop the image to the target dimensions
    cropped_image = image[crop_y:crop_y2, crop_x:crop_x2]

    # Calculate the position to paste the cropped image on the canvas
    paste_x = max(-(crop_x - x), 0)
    paste_y = max(-(crop_y - y), 0)

    canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    # Calculate the position to center the cropped image within the canvas
    center_x = paste_x + cropped_image.shape[1] // 2
    center_y = paste_y + cropped_image.shape[0] // 2
    canvas_center = np.array([canvas.shape[1] // 2, canvas.shape[0] // 2])

    # Calculate the position to paste the cropped image onto the canvas
    paste_position = canvas_center - np.array([center_x, center_y]) + np.array([paste_x, paste_y])

    # Create a black canvas with the target dimensions

    # Paste the cropped image onto the canvas
    canvas[paste_position[1]:paste_position[1] + cropped_image.shape[0],
        paste_position[0]:paste_position[0] + cropped_image.shape[1]] = cropped_image


    return canvas