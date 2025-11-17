from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI(title="AI 漫画线稿 API")

# 允许小程序跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_image(image_bytes):
    # 打开图片
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    # 转灰度
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (21,21), 0)
    sketch = cv2.divide(gray, blur, scale=255)
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    # 转回 base64
    pil_img = Image.fromarray(sketch_rgb)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.post("/process")
async def process(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img_base64 = process_image(img_bytes)
        return JSONResponse(content={"result": img_base64})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
