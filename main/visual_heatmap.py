# -*- coding: utf-8 -*-
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from MedMamba import VSSM as medmamba


MODEL_WEIGHT_PATH = "/mnt/ssd1/home/ailab/hlj/Passport/MedMamba/Pro_log_5_10/main/best_model.pth"
IMAGE_DIR = "/home/ailab/hlj/Passport/例图/3.Grad-CAM"
SAVE_DIR = "/mnt/ssd1/home/ailab/hlj/Passport/MedMamba/Grad"
SAVE_FIGURE_TEMPLATE = "/mnt/ssd1/home/ailab/hlj/Passport/MedMamba/Grad所有类别/{}_CNN_ConvMamba.jpg"

CATEGORY_CONFIGS = [
    ("engraving", 0, os.path.join(IMAGE_DIR, "engraving.png")),
    ("heat", 1, os.path.join(IMAGE_DIR, "heat.png")),
    ("ink", 2, os.path.join(IMAGE_DIR, "ink.png")),
    ("laser", 3, os.path.join(IMAGE_DIR, "laser.png")),
]


def build_model():
    return medmamba(
        depths=[2, 2, 4, 2],
        dims=[96, 192, 384, 768],
        num_classes=4,
    )


def get_target_layer(model, layer_index=-1, block_index=-1):
    """
    ConvMamba uses a sequence of MambaEncoder blocks. The last block gives
    the most class-specific token features for Grad-CAM.
    """
    return model.local_trans_pixel.layers[layer_index]


def tensor_to_chw(feature):
    if feature.dim() != 4:
        if feature.dim() == 3:
            batch_size, token_count, channels = feature.shape
            grid_size = int(np.sqrt(token_count))
            if grid_size * grid_size != token_count:
                raise ValueError(f"Expected a square token grid, got shape {tuple(feature.shape)}")
            return feature.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size).contiguous()
        raise ValueError(f"Expected 3D tokens or 4D feature map, got shape {tuple(feature.shape)}")

    # Some image models use NHWC. Convert to NCHW for Grad-CAM.
    if feature.shape[1] <= 16 and feature.shape[-1] > 16:
        return feature.permute(0, 3, 1, 2).contiguous()

    return feature


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activation = None
        self.gradient = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inputs, output):
        self.activation = tensor_to_chw(output.detach())

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradient = tensor_to_chw(grad_output[0].detach())

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def __call__(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activation = None
        self.gradient = None

        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        print(f"目标类别: {target_class}")
        print(f"模型输出: {output.detach().cpu()}")

        score = output[:, target_class].sum()
        score.backward()

        if self.activation is None:
            raise ValueError("未能通过 forward hook 获取到特征图")
        if self.gradient is None:
            raise ValueError("未能通过 backward hook 获取到梯度")

        print(f"特征图形状: {tuple(self.activation.shape)}")
        print(f"梯度形状: {tuple(self.gradient.shape)}")

        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activation, dim=1).squeeze(0)
        heatmap = torch.relu(heatmap)

        heatmap_np = heatmap.detach().cpu().numpy()
        heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)

        print(f"热图形状: {heatmap_np.shape}")
        print(f"归一化热图范围: [{heatmap_np.min():.4f}, {heatmap_np.max():.4f}]")

        return heatmap_np, output


def denormalize_image(input_tensor):
    image = input_tensor[0].detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    return np.clip(image, 0, 1)


def save_heatmap(input_tensor, heatmap, save_dir, category_name, save_image_path):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_image_path), exist_ok=True)

    original_img = denormalize_image(input_tensor)
    original_uint8 = (original_img * 255).astype(np.uint8)

    heatmap_resized = cv2.resize(heatmap, (input_tensor.shape[3], input_tensor.shape[2]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlayed = cv2.addWeighted(original_uint8, 0.7, heatmap_colored, 0.3, 0)

    overlay_path = os.path.join(save_dir, f"our_{category_name}_heatmap.png")
    raw_path = os.path.join(save_dir, f"our_{category_name}_heatmap_raw.png")

    cv2.imwrite(overlay_path, overlayed)
    cv2.imwrite(raw_path, heatmap_colored)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Overlayed")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_image_path, dpi=300)
    plt.show()

    print(f"热图已保存到: {overlay_path}")
    print(f"原始热图已保存到: {raw_path}")
    print(f"可视化结果已保存到: {save_image_path}")


def load_image(image_path, device):
    data_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    input_tensor = data_transform(image).unsqueeze(0).to(device)
    return input_tensor


if __name__ == "__main__":
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = build_model().to(device)
    print(f"加载模型权重: {MODEL_WEIGHT_PATH}")
    model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=device))
    model.eval()

    target_layer = get_target_layer(model, layer_index=-1, block_index=-1)
    grad_cam = GradCAM(model, target_layer)

    try:
        for category_name, target_class, image_path in CATEGORY_CONFIGS:
            print("\n" + "=" * 60)
            print(f"开始生成 {category_name} 的 Grad-CAM")
            print(f"输入图片: {image_path}")
            print(f"目标类别: {target_class}")

            input_tensor = load_image(image_path, device)
            print(f"输入张量形状: {tuple(input_tensor.shape)}")
            print(f"模型设备: {next(model.parameters()).device}")

            heatmap, output = grad_cam(input_tensor, target_class=target_class)
            save_image_path = SAVE_FIGURE_TEMPLATE.format(category_name)
            save_heatmap(input_tensor, heatmap, SAVE_DIR, category_name, save_image_path)
    finally:
        grad_cam.remove_hooks()
