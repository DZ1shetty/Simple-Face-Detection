import cv2
import numpy as np

def set_max_resolution(cap):
    """
    Sets the camera to the highest supported resolution.
    """
    # Try common high resolutions
    for width, height in [(1920,1080), (1280,720), (1024,768), (800,600)]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if actual_w >= width and actual_h >= height:
            break

def enhance_image(img):
    """
    Enhances the image for better display: sharpening, contrast, and color boost.
    """
    # Very light sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 4.5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(img, -1, kernel)
    
    # Light contrast enhancement (CLAHE)
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Very mild color boost
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s + 4, 0, 255)  # Very gentle saturation boost
    hsv_boosted = cv2.merge((h, s, v))
    vibrant = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)
    
    return vibrant
