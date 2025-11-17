# lineart.py
import cv2
import numpy as np

def lineart(img):
    """BGR 输入，返回 BGR 输出线稿（黑线白底）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blur, 60, 160)
    kernel = np.ones((2,2), np.uint8)
    dil = cv2.dilate(edges, kernel, iterations=1)
    line = 255 - dil
    line = cv2.cvtColor(line, cv2.COLOR_GRAY2BGR)
    return line
