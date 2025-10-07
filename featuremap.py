import os
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import torch
import torchvision.transforms as T
from transformers import AutoImageProcessor, AutoModel

def main(
    save_fg_mask=False,
    img_size=448,
    output_folder="outputs",
    pretrained_model_name="/mnt/afs/wusize/models/facebook/dinov3-vitb16-pretrain-lvd1689m",
    start_idx=1,
    end_idx=4
):
    os.makedirs(output_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --- load model ---
    print("Loading model:", pretrained_model_name)
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(pretrained_model_name).to(device)
    model.eval()

    # --- transforms ---
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
    ])

    # --- load images ---
    imgs = []
    for i in range(start_idx, end_idx+1):
        p = osp.join("images", f"{i}.jpg")
        if not osp.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))
    images = torch.stack(imgs, dim=0).to(device)  # B x C x H x W
    images_plot = ((images.clone().cpu().numpy()*0.5+0.5)*255).astype(np.uint8).transpose(0,2,3,1)

    with torch.no_grad():
        try:
            outputs = model(pixel_values=images)
        except TypeError:
            outputs = model(images)
        last_hidden = outputs.last_hidden_state  # [B, N, C]
        patch_tokens = last_hidden[:, 1:, :]  # 去掉 CLS token
        patch_tokens = torch.nn.functional.normalize(patch_tokens, p=2, dim=-1)
        x_norm_patchtokens = patch_tokens.cpu().numpy()  # B x N x C

    # --- 计算 patch 网格 ---
    B, N_minus1, C = x_norm_patchtokens.shape
    patch_h = patch_w = int(np.ceil(np.sqrt(N_minus1)))
    print(f"[Info] Token count = {N_minus1}, inferred patch grid = {patch_h}x{patch_w}")

    # pad 不足的 token
    pad = patch_h*patch_w - N_minus1
    if pad > 0:
        x_norm_patchtokens = np.pad(x_norm_patchtokens, ((0,0),(0,pad),(0,0)), mode='constant')

    # reshape
    feat_map = x_norm_patchtokens.reshape(B, patch_h, patch_w, C)

    # --- PCA 可视化 ---
    x_flat = feat_map.reshape(B*patch_h*patch_w, C)
    pca = PCA(n_components=3)
    pca_feats = pca.fit_transform(x_flat)
    pca_feats = minmax_scale(pca_feats).reshape(B, patch_h, patch_w, 3)

    for i in range(B):
        # 原图
        Image.fromarray(images_plot[i]).save(osp.join(output_folder, f"{i+1}_orig.png"))

        # 特征图 resize 到原图大小 (用 PIL)
        feat_img = (pca_feats[i]*255).astype(np.uint8)
        feat_resized = np.array(Image.fromarray(feat_img).resize((img_size, img_size), Image.BILINEAR))
        Image.fromarray(feat_resized).save(osp.join(output_folder, f"{i+1}_feat.png"))

        # 叠加图
        overlay = (0.5*images_plot[i] + 0.5*feat_resized).astype(np.uint8)
        Image.fromarray(overlay).save(osp.join(output_folder, f"{i+1}_overlay.png"))

    print(f"✅ Saved outputs to {output_folder}")


if __name__ == "__main__":
    main(
        save_fg_mask=True,
        img_size=448,
        output_folder="outputs",
        pretrained_model_name="/mnt/afs/wusize/models/facebook/dinov3-vitb16-pretrain-lvd1689m",
        start_idx=1,
        end_idx=4
    )
