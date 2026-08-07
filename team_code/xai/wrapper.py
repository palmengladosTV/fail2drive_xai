"""
Captum-compatible model wrappers for LidarCenterNet.

Each wrapper isolates a single output head so Captum can compute gradients
w.r.t. that specific prediction. The CaptumForwardAdapter handles the
translation between Captum's (inputs, additional_forward_args) convention
and the model's actual interface.
"""

import torch
from torch import nn


class TargetSpeedWrapper(nn.Module):
    """Wraps LidarCenterNet to expose target speed prediction."""

    def __init__(self, model, target_class_idx=None):
        super().__init__()
        self.model = model
        self.target_class_idx = target_class_idx

    def forward(self, rgb, lidar_bev, target_point, ego_vel, command):
        outputs = self.model(rgb, lidar_bev, target_point, ego_vel, command)
        pred_target_speed = outputs[1]
        if pred_target_speed is None:
            raise ValueError("Model did not produce target speed prediction. "
                             "Ensure use_controller_input_prediction=True in config.")
        if self.target_class_idx is not None:
            return pred_target_speed[:, self.target_class_idx]
        return pred_target_speed.max(dim=1).values


class WaypointWrapper(nn.Module):
    """Wraps LidarCenterNet to expose waypoint or checkpoint prediction."""

    def __init__(self, model, output_type='checkpoint', waypoint_idx=None):
        super().__init__()
        self.model = model
        self.output_type = output_type
        self.waypoint_idx = waypoint_idx

    def forward(self, rgb, lidar_bev, target_point, ego_vel, command):
        outputs = self.model(rgb, lidar_bev, target_point, ego_vel, command)
        if self.output_type == 'checkpoint':
            pred = outputs[2]
        else:
            pred = outputs[0]
        if pred is None:
            raise ValueError(f"Model did not produce {self.output_type} prediction.")
        if self.waypoint_idx is not None:
            return pred[:, self.waypoint_idx, :].norm(dim=1)
        return pred.norm(dim=2).sum(dim=1)


class BBoxWrapper(nn.Module):
    """Wraps LidarCenterNet to expose bounding box detection confidence."""

    def __init__(self, model, class_idx=0):
        super().__init__()
        self.model = model
        self.class_idx = class_idx

    def forward(self, rgb, lidar_bev, target_point, ego_vel, command):
        outputs = self.model(rgb, lidar_bev, target_point, ego_vel, command)
        pred_bb = outputs[6]
        if pred_bb is None:
            raise ValueError("Model did not produce bounding box prediction. "
                             "Ensure detect_boxes=True in config.")
        center_heatmap = pred_bb[0]
        return center_heatmap[:, self.class_idx].max(dim=2).values.max(dim=1).values


class SemanticWrapper(nn.Module):
    """Wraps LidarCenterNet to expose semantic segmentation logits."""

    def __init__(self, model, class_idx=0, spatial_pos=None, use_bev=False):
        super().__init__()
        self.model = model
        self.class_idx = class_idx
        self.spatial_pos = spatial_pos
        self.use_bev = use_bev

    def forward(self, rgb, lidar_bev, target_point, ego_vel, command):
        outputs = self.model(rgb, lidar_bev, target_point, ego_vel, command)
        if self.use_bev:
            pred = outputs[4]
        else:
            pred = outputs[3]
        if pred is None:
            raise ValueError("Model did not produce semantic prediction.")
        if self.spatial_pos is not None:
            h, w = self.spatial_pos
            return pred[:, self.class_idx, h, w]
        return pred[:, self.class_idx].mean(dim=(1, 2))


class CaptumForwardAdapter(nn.Module):
    """Adapts multi-input model for Captum's (inputs, additional_forward_args) convention.

    Captum passes attributed inputs as positional args, and non-attributed inputs
    via additional_forward_args. This adapter reconstructs the full call signature.

    Usage with Captum:
        adapter = CaptumForwardAdapter(wrapper, attributed_modalities=('rgb', 'lidar'))
        ig = IntegratedGradients(adapter)
        attrs = ig.attribute(
            inputs=(rgb, lidar_bev),
            additional_forward_args=(target_point, ego_vel, command),
            ...
        )
    """

    def __init__(self, output_wrapper, attributed_modalities=('rgb', 'lidar')):
        super().__init__()
        self.output_wrapper = output_wrapper
        self.attributed_modalities = attributed_modalities

    def forward(self, *args):
        if self.attributed_modalities == ('rgb', 'lidar'):
            rgb, lidar_bev = args[0], args[1]
            target_point, ego_vel, command = args[2], args[3], args[4]
        elif self.attributed_modalities == ('rgb',):
            rgb = args[0]
            lidar_bev, target_point, ego_vel, command = args[1], args[2], args[3], args[4]
        elif self.attributed_modalities == ('lidar',):
            lidar_bev = args[0]
            rgb, target_point, ego_vel, command = args[1], args[2], args[3], args[4]
        else:
            raise ValueError(f"Unknown attributed_modalities: {self.attributed_modalities}")
        return self.output_wrapper(rgb, lidar_bev, target_point, ego_vel, command)
