"""
Visualization for PlanT token-level attributions.

Since PlanT operates on object tokens (bounding boxes), not pixels,
the visualization shows which detected objects are most important
for the driving decision — rendered as colored boxes on a BEV canvas.
"""

import numpy as np
import cv2
from pathlib import Path
import torch


# PlanT2 type numbering (from plant2_variables.py class_nums)
TOKEN_TYPE_NAMES = {
    0: 'Padding',
    1: 'Car',
    2: 'Pedestrian',
    3: 'Static',
    4: 'StopSign',
    5: 'TrafLight',
    6: 'Emergency',
}
TOKEN_TYPE_COLORS = {
    0: (200, 200, 200),  # Padding: light gray
    1: (0, 165, 255),    # Car: orange (BGR)
    2: (0, 255, 0),      # Pedestrian: green
    3: (180, 180, 180),  # Static: gray
    4: (250, 160, 160),  # Stop sign: pink
    5: (0, 0, 255),      # Traffic light: red
    6: (255, 0, 255),    # Emergency: magenta
}


class PlanTXAIVisualizer:
    """Visualize token-level attributions for PlanT.

    Renders a BEV (Bird's Eye View) canvas with bounding boxes colored
    by their attribution importance. More important tokens are drawn with
    thicker borders and warmer colors.
    """

    def __init__(self, config):
        self.config = config

    def render_token_importance(self, bounding_boxes, token_importance,
                                 token_types=None, canvas_size=512, scale_factor=4):
        """Render BEV with boxes colored by importance.

        Args:
            bounding_boxes: (max_bbs, 8) tensor of detected objects
            token_importance: (max_bbs,) importance scores per token
            token_types: (max_bbs,) type index per token (optional)
            canvas_size: output image size in pixels
            scale_factor: pixels per meter

        Returns:
            BGR image (canvas_size, canvas_size, 3) uint8
        """
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 240

        if isinstance(bounding_boxes, torch.Tensor):
            bounding_boxes = bounding_boxes.detach().cpu().numpy()
        if isinstance(token_importance, torch.Tensor):
            token_importance = token_importance.detach().cpu().numpy()
        if token_types is not None and isinstance(token_types, torch.Tensor):
            token_types = token_types.detach().cpu().numpy()

        # Filter out padding tokens
        active_mask = np.abs(bounding_boxes).sum(axis=-1) > 0
        if not active_mask.any():
            return canvas

        # Normalize importance to [0, 1]
        imp = token_importance.copy()
        imp[~active_mask] = 0
        max_imp = imp.max()
        if max_imp > 0:
            imp_norm = imp / max_imp
        else:
            imp_norm = np.zeros_like(imp)

        origin = (canvas_size // 2, canvas_size // 2)
        ppm = self.config.pixels_per_meter * scale_factor

        # Draw ego vehicle
        ego_x = int(origin[0])
        ego_y = int(origin[1])
        cv2.circle(canvas, (ego_x, ego_y), radius=8, color=(0, 180, 0), thickness=-1)
        cv2.putText(canvas, 'EGO', (ego_x - 15, ego_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw bounding boxes with importance coloring
        for i in range(len(bounding_boxes)):
            if not active_mask[i]:
                continue

            box = bounding_boxes[i]
            # PlanT2 format: [type, x, y, yaw, speed, w, l] (type at index 0)
            # Old PlanT format: [x, y, ext_x, ext_y, yaw, speed, brake, type] (type at index 7)
            if len(box) == 7:
                obj_type = int(box[0])
                x, y = box[1], box[2]
            elif token_types is not None:
                obj_type = int(token_types[i])
                x, y = box[0], box[1]
            else:
                obj_type = int(box[7]) if len(box) > 7 else 0
                x, y = box[0], box[1]

            # Convert to pixel coordinates
            px = int(x * ppm + origin[0])
            py = int(y * ppm + origin[1])

            importance = imp_norm[i]

            # Color: interpolate from blue (low) to red (high importance)
            r = int(255 * importance)
            b = int(255 * (1 - importance))
            g = int(80 * (1 - importance))
            color = (b, g, r)

            # Size based on importance
            radius = int(6 + 14 * importance)
            thickness = max(1, int(3 * importance))

            cv2.circle(canvas, (px, py), radius=radius, color=color, thickness=thickness)

            # Label with type and importance
            type_name = TOKEN_TYPE_NAMES.get(obj_type, '?')
            label = f'{type_name} ({importance:.2f})'
            cv2.putText(canvas, label, (px + radius + 2, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

        return canvas

    def render_importance_bar(self, bounding_boxes, token_importance, token_types=None,
                               width=500, max_tokens=15):
        """Render a horizontal bar chart of token importances (top-N).

        Args:
            bounding_boxes: (max_bbs, 8) tensor
            token_importance: (max_bbs,) importance scores
            token_types: optional type labels
            width: chart width
            max_tokens: max number of tokens to show

        Returns:
            BGR image uint8
        """
        if isinstance(bounding_boxes, torch.Tensor):
            bounding_boxes = bounding_boxes.detach().cpu().numpy()
        if isinstance(token_importance, torch.Tensor):
            token_importance = token_importance.detach().cpu().numpy()

        active_mask = np.abs(bounding_boxes).sum(axis=-1) > 0
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            return np.ones((60, width, 3), dtype=np.uint8) * 255

        # Sort by importance
        imp_values = token_importance[active_indices]
        sorted_order = np.argsort(-imp_values)[:max_tokens]

        n_bars = len(sorted_order)
        bar_height = 20
        spacing = 5
        height = n_bars * (bar_height + spacing) + 40

        chart = np.ones((height, width, 3), dtype=np.uint8) * 255

        max_imp = imp_values.max() if imp_values.max() > 0 else 1.0

        cv2.putText(chart, 'Token Importance Ranking', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        for rank, sort_idx in enumerate(sorted_order):
            token_idx = active_indices[sort_idx]
            importance = token_importance[token_idx]
            norm_imp = importance / max_imp

            box = bounding_boxes[token_idx]
            if len(box) == 7:
                obj_type = int(box[0])
            elif len(box) > 7:
                obj_type = int(box[7])
            else:
                obj_type = 0
            type_name = TOKEN_TYPE_NAMES.get(obj_type, '?')
            type_color = TOKEN_TYPE_COLORS.get(obj_type, (128, 128, 128))

            y = 30 + rank * (bar_height + spacing)
            bar_w = int((width - 150) * norm_imp)

            cv2.rectangle(chart, (140, y), (140 + bar_w, y + bar_height), type_color, -1)

            label = f'#{token_idx} {type_name}'
            cv2.putText(chart, label, (5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(chart, f'{importance:.3f}', (145 + bar_w, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

        return chart

    def render_attention_matrix(self, attention_weights, token_types, num_object_tokens,
                                 layer_idx=-1, head_idx=None, canvas_size=400):
        """Render the self-attention matrix from PlanT's transformer.

        Shows how tokens attend to each other, with type labels.

        Args:
            attention_weights: list of (B, heads, seq, seq) per layer
            token_types: (seq_len,) type of each token
            num_object_tokens: number of actual object tokens (before route)
            layer_idx: which layer to visualize (-1 = last)
            head_idx: which head (None = average over heads)

        Returns:
            BGR image uint8
        """
        attn = attention_weights[layer_idx]
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().cpu().numpy()
        if isinstance(token_types, torch.Tensor):
            token_types = token_types.detach().cpu().numpy()

        # Take first batch element
        if attn.ndim == 4:
            attn = attn[0]
        if token_types.ndim == 2:
            token_types = token_types[0]

        # Average or select head
        if head_idx is not None:
            attn_map = attn[head_idx]
        else:
            attn_map = attn.mean(axis=0)

        # Normalize for visualization
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # Resize to canvas
        attn_vis = cv2.resize(attn_map.astype(np.float32),
                              (canvas_size, canvas_size),
                              interpolation=cv2.INTER_NEAREST)
        attn_colored = cv2.applyColorMap((attn_vis * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

        # Add token type labels on axes
        seq_len = len(token_types)
        step = canvas_size / seq_len

        for i in range(min(seq_len, 30)):
            t = int(token_types[i])
            name = TOKEN_TYPE_NAMES.get(t, '?')[:3]
            pos = int(i * step + step / 2)
            cv2.putText(attn_colored, name, (pos - 5, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

        return attn_colored

    def render_combined(self, bounding_boxes, token_importance, token_types=None,
                         attention_data=None, method_name='', output_head='',
                         save_path=None, step=None):
        """Generate a full PlanT XAI panel.

        Args:
            bounding_boxes: (max_bbs, 8) or (B, max_bbs, 8)
            token_importance: (max_bbs,) or (B, max_bbs)
            attention_data: optional dict from compute_attention()
            save_path: directory to save
            step: frame number

        Returns:
            Combined BGR image, or None if saved to disk
        """
        if isinstance(bounding_boxes, torch.Tensor) and bounding_boxes.dim() == 3:
            bounding_boxes = bounding_boxes[0]
        if isinstance(token_importance, torch.Tensor) and token_importance.dim() == 2:
            token_importance = token_importance[0]

        panels = []

        # BEV with importance
        bev_panel = self.render_token_importance(bounding_boxes, token_importance,
                                                  token_types)
        panels.append(bev_panel)

        # Bar chart
        bar_panel = self.render_importance_bar(bounding_boxes, token_importance,
                                               token_types, width=bev_panel.shape[1])
        panels.append(bar_panel)

        # Attention matrix (if available)
        if attention_data and 'attention_weights' in attention_data:
            attn_panel = self.render_attention_matrix(
                attention_data['attention_weights'],
                attention_data['token_types'],
                attention_data['num_object_tokens'],
                canvas_size=bev_panel.shape[1])
            panels.append(attn_panel)

        combined = np.concatenate(panels, axis=0)

        # Title
        title = f'PlanT XAI: {method_name} -> {output_head}'
        cv2.putText(combined, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 1, cv2.LINE_AA)

        if save_path is not None:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f'plant_{method_name}_{output_head}'
            if step is not None:
                filename += f'_{step:04d}'
            filename += '.png'
            cv2.imwrite(str(save_dir / filename), combined)
            return None

        return combined