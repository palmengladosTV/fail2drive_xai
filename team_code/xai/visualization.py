"""
XAI Visualization module.

Generates attribution heatmaps overlaid on RGB images and LiDAR BEV,
following the visual style of LidarCenterNet.visualize_model().
"""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image

import torch


def _normalize_attribution(attr, sign='absolute', percentile=99):
    """Normalize attribution tensor to [0, 1] range for visualization.

    Args:
        attr: Attribution tensor (C, H, W) or (H, W)
        sign: 'positive', 'negative', 'absolute', or 'all'
        percentile: Clip at this percentile to reduce outlier influence
    """
    if attr.dim() == 3:
        attr = attr.sum(dim=0)

    attr = attr.cpu().numpy().astype(np.float32)

    if sign == 'positive':
        attr = np.maximum(attr, 0)
    elif sign == 'negative':
        attr = np.maximum(-attr, 0)
    elif sign == 'absolute':
        attr = np.abs(attr)

    if attr.max() == 0:
        return np.zeros_like(attr)

    vmax = np.percentile(attr, percentile)
    if vmax > 0:
        attr = np.clip(attr / vmax, 0, 1)
    return attr


def _apply_colormap(heatmap, colormap=cv2.COLORMAP_JET):
    """Convert a single-channel [0,1] heatmap to a BGR color image."""
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    return colored


def _overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay a heatmap on an image with alpha blending.

    Args:
        image: BGR image (H, W, 3) uint8
        heatmap: Single channel [0, 1] float (H, W)
        alpha: Blending factor for the heatmap
    """
    colored = _apply_colormap(heatmap, colormap)
    mask = (heatmap > 0.01).astype(np.float32)[:, :, None]
    blended = image.astype(np.float32) * (1 - alpha * mask) + colored.astype(np.float32) * alpha * mask
    return np.clip(blended, 0, 255).astype(np.uint8)


class XAIVisualizer:
    """Generate visual explanations as heatmap overlays.

    Produces panels matching the layout of LidarCenterNet.visualize_model():
    RGB image on top, BEV on bottom, with attribution heatmaps overlaid.
    """

    def __init__(self, config):
        self.config = config

    def render_rgb_attribution(self, rgb_tensor, attribution, sign='absolute',
                                alpha=0.5, colormap=cv2.COLORMAP_JET):
        """Render attribution heatmap on RGB image.

        Args:
            rgb_tensor: Normalized RGB tensor (3, H, W) — will be denormalized for display
            attribution: Attribution tensor (3, H, W) or (H, W)
            sign: How to handle sign of attributions
            alpha: Overlay opacity

        Returns:
            BGR image (H, W, 3) uint8
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_np = rgb_tensor.cpu().numpy().transpose(1, 2, 0)
        rgb_np = (rgb_np * std + mean) * 255
        rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

        heatmap = _normalize_attribution(attribution, sign=sign)
        if heatmap.shape != rgb_bgr.shape[:2]:
            heatmap = cv2.resize(heatmap, (rgb_bgr.shape[1], rgb_bgr.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        return _overlay_heatmap(rgb_bgr, heatmap, alpha=alpha, colormap=colormap)

    def render_lidar_attribution(self, lidar_bev_tensor, attribution, sign='absolute',
                                  alpha=0.6, colormap=cv2.COLORMAP_JET, scale_factor=4):
        """Render attribution heatmap on LiDAR BEV visualization.

        Args:
            lidar_bev_tensor: LiDAR BEV tensor (C, H, W)
            attribution: Attribution tensor (C, H, W) or (H, W)
            scale_factor: Upscale factor for display (matches visualize_model)

        Returns:
            BGR image (H*scale, W*scale, 3) uint8
        """
        lidar_np = lidar_bev_tensor.cpu().numpy()[0]
        lidar_gray = 255 - (lidar_np * 255).astype(np.uint8)
        lidar_bgr = np.stack([lidar_gray, lidar_gray, lidar_gray], axis=-1)
        lidar_bgr = cv2.resize(lidar_bgr,
                                dsize=(lidar_bgr.shape[1] * scale_factor,
                                       lidar_bgr.shape[0] * scale_factor),
                                interpolation=cv2.INTER_NEAREST)

        heatmap = _normalize_attribution(attribution, sign=sign)
        heatmap = cv2.resize(heatmap,
                              dsize=(lidar_bgr.shape[1], lidar_bgr.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

        return _overlay_heatmap(lidar_bgr, heatmap, alpha=alpha, colormap=colormap)

    def render_modality_importance(self, attributions, width=400, height=60):
        """Render a bar chart showing relative importance of each attributed modality.

        Args:
            attributions: dict {modality_name: attribution_tensor}

        Returns:
            BGR image (height, width, 3) uint8
        """
        importances = {}
        for name, attr in attributions.items():
            val = attr.abs().sum().item()
            if val > 0:
                importances[name] = val

        if not importances:
            importances = {list(attributions.keys())[0]: 1.0}

        total = sum(importances.values()) + 1e-8
        for name in importances:
            importances[name] /= total

        bar_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        colors = {
            'rgb': (0, 120, 255),
            'lidar': (0, 200, 0),
        }

        x_offset = 10
        bar_height = 20
        y_offset = 10

        for i, (name, importance) in enumerate(importances.items()):
            color = colors.get(name, (128, 128, 128))
            bar_width = int((width - 20) * importance)
            y = y_offset + i * (bar_height + 5)
            cv2.rectangle(bar_img, (x_offset, y), (x_offset + bar_width, y + bar_height), color, -1)
            cv2.putText(bar_img, f'{name}: {importance:.1%}', (x_offset + bar_width + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        return bar_img

    def render_combined(self, rgb_tensor, lidar_bev_tensor, attributions,
                         method_name='', output_head='',
                         pred_target_speed=None, target_speeds=None,
                         save_path=None, step=None):
        """Generate a full XAI panel with attribution overlays.

        Args:
            rgb_tensor: (3, H, W) normalized RGB
            lidar_bev_tensor: (C, H, W) LiDAR BEV
            attributions: dict with 'rgb' and/or 'lidar' attribution tensors
            method_name: Name of the XAI method (for title)
            output_head: Name of the output being explained
            pred_target_speed: Optional speed prediction for overlay
            target_speeds: Optional list of target speed values
            save_path: Directory to save the panel
            step: Step number for filename

        Returns:
            Combined BGR image if save_path is None, else None
        """
        panels = []

        if 'rgb' in attributions:
            rgb_panel = self.render_rgb_attribution(rgb_tensor, attributions['rgb'])
            panels.append(rgb_panel)
        else:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            rgb_np = rgb_tensor.cpu().numpy().transpose(1, 2, 0)
            rgb_np = (rgb_np * std + mean) * 255
            rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
            panels.append(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        if 'lidar' in attributions and attributions['lidar'].abs().sum() > 0:
            lidar_panel = self.render_lidar_attribution(lidar_bev_tensor, attributions['lidar'])
            target_width = panels[0].shape[1] if panels else lidar_panel.shape[1]
            lidar_panel = cv2.resize(lidar_panel, (target_width, target_width),
                                      interpolation=cv2.INTER_NEAREST)
            panels.append(lidar_panel)

        importance_bar = self.render_modality_importance(attributions, width=panels[0].shape[1])
        panels.append(importance_bar)

        combined = np.concatenate(panels, axis=0)

        title = f'XAI: {method_name} -> {output_head}'
        cv2.putText(combined, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        if pred_target_speed is not None and target_speeds is not None:
            speed_probs = torch.softmax(pred_target_speed, dim=0).cpu().numpy()
            pred_idx = speed_probs.argmax()
            speed_text = f'Pred speed: {target_speeds[pred_idx]:.1f} m/s (p={speed_probs[pred_idx]:.2f})'
            cv2.putText(combined, speed_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if save_path is not None:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f'{method_name}_{output_head}'
            if step is not None:
                filename += f'_{step:04d}'
            filename += '.png'
            cv2.imwrite(str(save_dir / filename), combined)
            return None

        return combined

    def render_attention_flow(self, attention_weights, image_shape, lidar_shape,
                               save_path=None, step=None):
        """Visualize cross-modal attention between image and LiDAR tokens.

        Args:
            attention_weights: List of attention weight arrays from GPT fusion blocks
            image_shape: (H, W) of the input image
            lidar_shape: (H, W) of the LiDAR BEV

        Returns:
            BGR image showing attention flow, or None if saved to disk
        """
        if not attention_weights:
            return None

        num_scales = len(attention_weights)
        panel_size = 256
        combined_width = num_scales * panel_size
        combined = np.ones((panel_size, combined_width, 3), dtype=np.uint8) * 255

        for i, attn in enumerate(attention_weights):
            if isinstance(attn, torch.Tensor):
                attn = attn.cpu().numpy()
            while attn.ndim > 2:
                attn = attn.mean(axis=0)

            attn = attn.astype(np.float32)
            attn = np.log(attn + 1e-8)
            vmin = np.percentile(attn, 2)
            vmax = np.percentile(attn, 98)
            attn_norm = np.clip((attn - vmin) / (vmax - vmin + 1e-8), 0, 1)

            attn_resized = cv2.resize(attn_norm, (panel_size, panel_size),
                                       interpolation=cv2.INTER_LINEAR)
            colored = _apply_colormap(attn_resized)
            x_start = i * panel_size
            combined[:, x_start:x_start + panel_size] = colored

            cv2.putText(combined, f'Scale {i}', (x_start + 5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if save_path is not None:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = 'attention_flow'
            if step is not None:
                filename += f'_{step:04d}'
            filename += '.png'
            cv2.imwrite(str(save_dir / filename), combined)
            return None

        return combined
