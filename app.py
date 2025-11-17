# app.py
import os
import base64
import cv2
import numpy as np
import gradio as gr
from lineart import lineart
from sketch import pencil
from comic import comic_effect
from animegan import apply_anime

PORT = int(os.environ.get("PORT", 8080))

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def to_base64_image(img_bgr):
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode()

def from_pil_or_np(img):
    # Gradio 给的是 PIL 图或 numpy RGB
    if isinstance(img, np.ndarray):
        return rgb_to_bgr(img)
    else:
        arr = np.array(img)
        return rgb_to_bgr(arr)

def process(img, style):
    if img is None:
        return None

    bgr = from_pil_or_np(img)

    if style == "纯净线稿":
        out = lineart(bgr)
    elif style == "漫画线稿":
        out = comic_effect(bgr)
    elif style == "素描风":
        out = pencil(bgr)
    elif style == "二次元 AnimeGAN":
        out = apply_anime(bgr)
    else:
        out = lineart(bgr)

    return bgr_to_rgb(out)

title = "AI 漫画线稿（Render 部署示例）"
desc = "上传照片 → 选择风格 → 点击转换（纯线稿 / 漫画线稿 / 素描 / 二次元）"

demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Image(type="pil", label="上传图片"),
        gr.Dropdown(["纯净线稿", "漫画线稿", "素描风", "二次元 AnimeGAN"], label="选择风格")
    ],
    outputs=gr.Image(type="numpy", label="输出"),
    title=title,
    description=desc,
    allow_flagging="never"
)

if __name__ == "__main__":
    # Render 会提供 PORT 环境变量
    demo.launch(server_name="0.0.0.0", server_port=PORT)
