# sketch.py
import cv2
import numpy as np

def pencil(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21,21), 0)
    # 防止除0，使用微小常数
    sketch = cv2.divide(gray, 255 - blur + 1e-6, scale=256)
    sketch_color = cv2.cvtColor(sketch.astype('uint8'), cv2.COLOR_GRAY2BGR)
    return sketch_color
