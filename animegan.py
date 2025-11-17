# animegan.py
import cv2, numpy as np, os, requests, base64

HF_TOKEN = os.environ.get("HF_API_TOKEN", "")  # 可选：若你有 HF token 可用更强模型

def local_anime_style(img):
    # 快速本地近似二次元风：双边滤波 + 颜色量化
    img_small = cv2.pyrDown(img)
    for _ in range(2):
        img_small = cv2.bilateralFilter(img_small, d=9, sigmaColor=75, sigmaSpace=75)
    img_up = cv2.pyrUp(img_small)
    Z = img_up.reshape((-1,3)).astype(np.float32)
    K = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        _,label,center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()].reshape(img_up.shape)
        return res
    except Exception:
        return img

def apply_anime(img):
    # 优先使用 HF 推理（如果配置了），否则使用本地快速方法
    if HF_TOKEN:
        url = "https://api-inference.huggingface.co/models/akhaliq/animegan3-shinkai-512"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        _, buf = cv2.imencode(".png", img)
        b64 = base64.b64encode(buf.tobytes()).decode()
        data = {"inputs": b64}
        try:
            r = requests.post(url, headers=headers, json=data, timeout=60)
            if r.status_code == 200:
                j = r.json()
                # HF 返回可能在不同字段，尝试常见路径
                out_b64 = None
                if isinstance(j, dict):
                    out_b64 = j.get("image_base64") or j.get("data")
                if out_b64:
                    out = base64.b64decode(out_b64)
                    arr = np.frombuffer(out, dtype=np.uint8)
                    img2 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    return img2
        except Exception:
            pass
    return local_anime_style(img)
