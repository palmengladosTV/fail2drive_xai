"""
Attribution computation engine using Captum.

Provides a unified interface to compute feature attributions for any output head
of LidarCenterNet using multiple XAI methods.
"""

import torch
from torch import nn
from captum.attr import (
    Saliency,
    IntegratedGradients,
    LayerGradCam,
    FeatureAblation,
    NoiseTunnel,
    DeepLift,
)

from xai.wrapper import (
    TargetSpeedWrapper,
    WaypointWrapper,
    BBoxWrapper,
    SemanticWrapper,
    CaptumForwardAdapter,
)


def _unwrap_model(model):
    """Get the raw model from DDP or torch.compile wrappers."""
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model


def _get_layer(model, layer_name):
    """Resolve a dot-separated layer name to the actual module."""
    parts = layer_name.split('.')
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


class XAIEngine:
    """Central engine for computing attributions on LidarCenterNet.

    Args:
        model: LidarCenterNet instance (or DDP/compiled wrapped version)
        config: GlobalConfig instance
        device: torch device
    """

    def __init__(self, model, config, device='cuda'):
        self.raw_model = _unwrap_model(model)
        self.config = config
        self.device = device
        self._prepare_model_for_attribution()

    def _prepare_model_for_attribution(self):
        """Set model to train mode (needed for cuDNN RNN backward) but freeze BatchNorm."""
        self.raw_model.train()
        for module in self.raw_model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                   torch.nn.SyncBatchNorm, torch.nn.LayerNorm)):
                module.eval()

    def _build_wrapper(self, output_head, **kwargs):
        """Build the appropriate output wrapper for the given head."""
        if output_head == 'target_speed':
            return TargetSpeedWrapper(self.raw_model, target_class_idx=kwargs.get('class_idx'))
        elif output_head == 'checkpoint':
            return WaypointWrapper(self.raw_model, output_type='checkpoint',
                                   waypoint_idx=kwargs.get('waypoint_idx'))
        elif output_head == 'waypoint':
            return WaypointWrapper(self.raw_model, output_type='waypoint',
                                   waypoint_idx=kwargs.get('waypoint_idx'))
        elif output_head == 'bbox':
            return BBoxWrapper(self.raw_model, class_idx=kwargs.get('class_idx', 0))
        elif output_head == 'semantic':
            return SemanticWrapper(self.raw_model, class_idx=kwargs.get('class_idx', 0),
                                   spatial_pos=kwargs.get('spatial_pos'),
                                   use_bev=kwargs.get('use_bev', False))
        else:
            raise ValueError(f"Unknown output_head: {output_head}")

    def compute_attribution(self, method, output_head, rgb, lidar_bev,
                            target_point, ego_vel, command,
                            attributed_modalities=('rgb', 'lidar'),
                            smooth=False, smooth_samples=10,
                            **kwargs):
        """Compute attribution for a given method and output head.

        Args:
            method: One of 'saliency', 'integrated_gradients', 'grad_cam', 'feature_ablation', 'deeplift'
            output_head: One of 'target_speed', 'checkpoint', 'waypoint', 'bbox', 'semantic'
            rgb: Input RGB tensor (B, 3, H, W)
            lidar_bev: Input LiDAR BEV tensor (B, C, H, W)
            target_point: Target point tensor (B, 2)
            ego_vel: Velocity tensor (B, 1)
            command: Command tensor (B, 6)
            attributed_modalities: Tuple of modalities to attribute ('rgb', 'lidar', or both)
            smooth: Whether to use SmoothGrad (NoiseTunnel) on top
            smooth_samples: Number of samples for SmoothGrad
            **kwargs: Additional args passed to wrapper (class_idx, waypoint_idx, etc.)

        Returns:
            dict with keys matching attributed_modalities, values are attribution tensors
        """
        wrapper = self._build_wrapper(output_head, **kwargs)
        adapter = CaptumForwardAdapter(wrapper, attributed_modalities=attributed_modalities)

        rgb = rgb.detach().requires_grad_(True)
        lidar_bev = lidar_bev.detach().requires_grad_(True)

        if attributed_modalities == ('rgb', 'lidar'):
            inputs = (rgb, lidar_bev)
            additional_forward_args = (target_point, ego_vel, command)
            baselines = (torch.zeros_like(rgb), torch.zeros_like(lidar_bev))
        elif attributed_modalities == ('rgb',):
            inputs = (rgb,)
            additional_forward_args = (lidar_bev, target_point, ego_vel, command)
            baselines = (torch.zeros_like(rgb),)
        elif attributed_modalities == ('lidar',):
            inputs = (lidar_bev,)
            additional_forward_args = (rgb, target_point, ego_vel, command)
            baselines = (torch.zeros_like(lidar_bev),)
        else:
            raise ValueError(f"Unknown attributed_modalities: {attributed_modalities}")

        with torch.enable_grad():
            if method == 'saliency':
                attr_method = Saliency(adapter)
                if smooth:
                    attr_method = NoiseTunnel(attr_method)
                    attrs = attr_method.attribute(inputs,
                                                  additional_forward_args=additional_forward_args,
                                                  nt_samples=smooth_samples,
                                                  nt_type='smoothgrad')
                else:
                    attrs = attr_method.attribute(inputs,
                                                  additional_forward_args=additional_forward_args)

            elif method == 'integrated_gradients':
                attr_method = IntegratedGradients(adapter)
                n_steps = kwargs.get('n_steps', self.config.xai_n_steps)
                if smooth:
                    attr_method = NoiseTunnel(attr_method)
                    attrs = attr_method.attribute(inputs, baselines=baselines,
                                                  additional_forward_args=additional_forward_args,
                                                  nt_samples=smooth_samples,
                                                  nt_type='smoothgrad',
                                                  internal_batch_size=1)
                else:
                    attrs = attr_method.attribute(inputs, baselines=baselines,
                                                  additional_forward_args=additional_forward_args,
                                                  n_steps=n_steps,
                                                  internal_batch_size=1)

            elif method == 'grad_cam':
                layer_name = kwargs.get('layer_name', self._default_gradcam_layer())
                layer = _get_layer(self.raw_model, layer_name)
                attr_method = LayerGradCam(adapter, layer)
                # GradCAM returns a single spatial map for the target layer.
                # We attribute to the first input only to avoid shape mismatches,
                # then upsample and assign the result to the appropriate modality.
                first_input = inputs[0] if isinstance(inputs, tuple) else inputs
                grad_cam_attr = attr_method.attribute(
                    first_input,
                    additional_forward_args=(inputs[1],) + additional_forward_args if len(inputs) > 1
                    else additional_forward_args)
                # Upsample GradCAM to match first input spatial dimensions
                import torch.nn.functional as F_interp
                if grad_cam_attr.dim() == 4 and grad_cam_attr.shape[2:] != first_input.shape[2:]:
                    grad_cam_attr = F_interp.interpolate(
                        grad_cam_attr, size=first_input.shape[2:], mode='bilinear', align_corners=False)
                # Build result dict: GradCAM goes to first modality, zeros for others
                result = {}
                result[attributed_modalities[0]] = grad_cam_attr.detach()
                for mod in attributed_modalities[1:]:
                    matching_input = lidar_bev if mod == 'lidar' else rgb
                    result[mod] = torch.zeros_like(matching_input)
                return result

            elif method == 'feature_ablation':
                attr_method = FeatureAblation(adapter)
                feature_mask = self._build_feature_mask(inputs, attributed_modalities)
                attrs = attr_method.attribute(inputs,
                                              additional_forward_args=additional_forward_args,
                                              feature_mask=feature_mask)

            elif method == 'deeplift':
                attr_method = DeepLift(adapter)
                if smooth:
                    attr_method = NoiseTunnel(attr_method)
                    attrs = attr_method.attribute(inputs, baselines=baselines,
                                                  additional_forward_args=additional_forward_args,
                                                  nt_samples=smooth_samples,
                                                  nt_type='smoothgrad')
                else:
                    attrs = attr_method.attribute(inputs, baselines=baselines,
                                                  additional_forward_args=additional_forward_args)
            else:
                raise ValueError(f"Unknown method: {method}. "
                                 "Options: saliency, integrated_gradients, grad_cam, feature_ablation, deeplift")

        if not isinstance(attrs, tuple):
            attrs = (attrs,)

        result = {}
        for modality, attr in zip(attributed_modalities, attrs):
            result[modality] = attr.detach()

        return result

    def compute_attention(self, rgb, lidar_bev, target_point, ego_vel, command):
        """Extract attention weights from the GPT fusion blocks and transformer decoder.

        Returns:
            dict with 'fusion_attention' (list of per-scale attention maps)
            and 'decoder_attention' if tp_attention is enabled
        """
        result = {}

        if hasattr(self.raw_model, 'backbone') and hasattr(self.raw_model.backbone, 'set_store_attention'):
            self.raw_model.backbone.set_store_attention(True)

        with torch.no_grad():
            outputs = self.raw_model(rgb, lidar_bev, target_point, ego_vel, command)
            if outputs[7] is not None:
                result['decoder_attention'] = outputs[7]

        if hasattr(self.raw_model, 'backbone') and hasattr(self.raw_model.backbone, 'get_attention_maps'):
            attention_maps = self.raw_model.backbone.get_attention_maps()
            if attention_maps:
                result['fusion_attention'] = attention_maps
            self.raw_model.backbone.set_store_attention(False)

        return result

    def _default_gradcam_layer(self):
        """Return the default layer for GradCAM by finding the last feature-producing module."""
        backbone = self.raw_model.backbone
        encoder = getattr(backbone, 'image_encoder', None)
        if encoder is None:
            return 'backbone'

        # timm FeatureListNet uses return_layers dict: {'s1': '0', 's2': '1', ...}
        if hasattr(encoder, 'return_layers'):
            last_layer_name = list(encoder.return_layers.keys())[-1]
            layer = getattr(encoder, last_layer_name, None)
            if layer is not None:
                return f'backbone.image_encoder.{last_layer_name}'

        # Fallback: try standard ResNet layer names
        for name in ('layer4', 'layer3', 'features'):
            if hasattr(encoder, name):
                return f'backbone.image_encoder.{name}'

        return 'backbone'

    def _build_feature_mask(self, inputs, attributed_modalities):
        """Build a feature mask for FeatureAblation that groups by spatial regions.

        Divides each input into a grid of regions for coarser attribution.
        """
        masks = []
        region_idx = 0
        grid_h, grid_w = 8, 8

        for inp in inputs:
            b, c, h, w = inp.shape
            mask = torch.zeros(1, 1, h, w, device=inp.device, dtype=torch.long)
            cell_h = h // grid_h
            cell_w = w // grid_w
            for i in range(grid_h):
                for j in range(grid_w):
                    mask[0, 0, i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w] = region_idx
                    region_idx += 1
            mask = mask.expand(b, c, h, w)
            masks.append(mask)

        return tuple(masks)
