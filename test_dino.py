import os
import torch
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

# ==============================
# 配置
# ==============================
pretrained_model_name = "/mnt/afs/wusize/models/facebook/dinov3-vitb16-pretrain-lvd1689m"
output_dir = "/mnt/afs/wusize/projects/dinov3/output"
os.makedirs(output_dir, exist_ok=True)
img_size = 224   # 原图 resize 尺寸
upscale = 16     # 特征图放大倍数

# 图片 URL 列表
image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
]

# ==============================
# 加载模型
# ==============================
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")

# ==============================
# 遍历图片
# ==============================
for idx, url in enumerate(image_urls):
    print(f"\nProcessing {idx+1}/{len(image_urls)}: {url}")
    image = load_image(url)
    orig_img = image.resize((img_size, img_size))

    # 保存原图
    orig_save_path = os.path.join(output_dir, f"image_{idx+1}_orig.png")
    orig_img.save(orig_save_path)

    # 前向推理
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)

    # dense features
    dense_features = outputs.last_hidden_state[:, 1:, :]  # 去掉 CLS token
    B, N, C = dense_features.shape
    H = W = int(N ** 0.5)
    dense_features = dense_features[:, :H*W, :].reshape(B, H, W, C)
    dense_np = dense_features.squeeze(0).detach().cpu().numpy()
    dense_flat = dense_np.reshape(-1, C)  # (N, C)

    # PCA 降到 3 维
    pca = PCA(n_components=3)
    rgb_proj = pca.fit_transform(dense_flat)  # (N,3)
    # 归一化到 [0,255]
    rgb_min, rgb_max = rgb_proj.min(0), rgb_proj.max(0)
    rgb_norm = (rgb_proj - rgb_min) / (rgb_max - rgb_min)
    rgb_uint8 = (rgb_norm * 255).astype(np.uint8)
    # 重建成 H x W x 3
    rgb_image = rgb_uint8.reshape(H, W, 3)
    # 放大到更高分辨率
    rgb_image = Image.fromarray(rgb_image).resize((H*upscale, W*upscale), Image.BILINEAR)

    heatmap_path = os.path.join(output_dir, f"image_{idx+1}_featuremap.png")
    rgb_image.save(heatmap_path)

    print(f"✅ Saved original and PCA feature map: {orig_save_path}, {heatmap_path}")



