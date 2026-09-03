"""
Standalone XAI analysis script for post-hoc explanation of saved checkpoints.

Works with tensor snapshots (.pt files) saved during evaluation when XAI_ENABLED=1.

Usage:
    # Analyse gespeicherte Evaluations-Tensoren (empfohlen):
    python -m xai.analysis \
        --checkpoint ./checkpoints/tfpp \
        --tensor_dir ./eval_output/<route>/xai_tensors \
        --output_dir ./xai_results \
        --methods saliency integrated_gradients \
        --output_heads target_speed checkpoint

    # Analyse aller .pt Dateien in einem Verzeichnis:
    python -m xai.analysis \
        --checkpoint ./checkpoints/tfpp \
        --tensor_dir ./eval_output/<route>/xai_tensors \
        --output_dir ./xai_results \
        --methods saliency integrated_gradients grad_cam \
        --output_heads target_speed checkpoint waypoint \
        --num_samples 20
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GlobalConfig
from model import LidarCenterNet

from xai.attributions import XAIEngine
from xai.visualization import XAIVisualizer

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

jsonpickle_numpy.register_handlers()


def detect_model_type(checkpoint_path):
    """Detect whether checkpoint is TF++ (.pth) or PlanT2 (.ckpt)."""
    pth_files = [f for f in os.listdir(checkpoint_path)
                 if f.endswith('.pth') and f.startswith('model')]
    ckpt_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')]

    if pth_files:
        return 'tfpp'
    elif ckpt_files:
        return 'plant2'
    else:
        raise FileNotFoundError(
            f"No model files found in {checkpoint_path}. "
            "Expected model_*.pth (TF++) or *.ckpt (PlanT2).")


def load_model(checkpoint_path, device):
    """Load model from a checkpoint directory. Auto-detects TF++ vs PlanT2."""
    model_type = detect_model_type(checkpoint_path)

    if model_type == 'tfpp':
        return _load_tfpp(checkpoint_path, device)
    else:
        return _load_plant2(checkpoint_path, device)


def _load_tfpp(checkpoint_path, device):
    """Load a LidarCenterNet (TF++) model."""
    config_file = os.path.join(checkpoint_path, 'config.json')
    with open(config_file, 'rt', encoding='utf-8') as f:
        json_config = f.read()

    loaded_config = jsonpickle.decode(json_config)
    config = GlobalConfig()
    config.__dict__.update(loaded_config.__dict__)

    model = LidarCenterNet(config)
    if config.sync_batch_norm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_files = sorted([f for f in os.listdir(checkpoint_path)
                          if f.endswith('.pth') and f.startswith('model')])
    if not model_files:
        raise FileNotFoundError(f"No model .pth files found in {checkpoint_path}")

    print(f"Loading TF++ model: {model_files[0]}")
    state_dict = torch.load(os.path.join(checkpoint_path, model_files[0]), map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, config


def _load_plant2(checkpoint_path, device):
    """Load a PlanT2 (HFLM) model from a Lightning checkpoint."""
    from plant2_lit_module import LitHFLM

    ckpt_files = sorted([f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')])
    ckpt_path = os.path.join(checkpoint_path, ckpt_files[0])

    print(f"Loading PlanT2 model: {ckpt_files[0]}")
    lit_model = LitHFLM.load_from_checkpoint(ckpt_path, map_location=device, strict=False)
    lit_model.eval()

    config = GlobalConfig()
    return lit_model, config


def load_tensor_samples(tensor_dir, num_samples=None):
    """Load saved .pt tensor files from an evaluation run.

    These are saved by sensor_agent.py when XAI_ENABLED=1.
    Each .pt file contains: rgb, lidar_bev, target_point, ego_vel, command, step
    """
    tensor_dir = Path(tensor_dir)
    pt_files = sorted(tensor_dir.glob('*.pt'))

    if not pt_files:
        raise FileNotFoundError(
            f"No .pt files found in {tensor_dir}.\n"
            f"Run evaluation with XAI_ENABLED=1 SAVE_PATH=<path> to generate them."
        )

    if num_samples is not None:
        pt_files = pt_files[:num_samples]

    print(f"Found {len(pt_files)} tensor snapshots in {tensor_dir}")
    return pt_files


def run_analysis(args):
    """Run XAI analysis on saved tensor snapshots."""
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    print(f'Loading model from: {args.checkpoint}')
    model_type = detect_model_type(args.checkpoint)
    model, config = load_model(args.checkpoint, device)

    from datetime import datetime
    run_name = getattr(args, 'run_name', None)
    if run_name:
        base_dir = Path(args.output_dir) / run_name
    else:
        base_dir = Path(args.output_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')

    if getattr(args, 'degrade', None):
        output_dir = base_dir / f'degrade_{args.degrade}_{args.degrade_method}_{args.degrade_strength}'
    elif run_name:
        output_dir = base_dir / 'reference'
    else:
        output_dir = base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = load_tensor_samples(args.tensor_dir, args.num_samples)

    # Detect tensor format from first sample
    first_sample = torch.load(pt_files[0], map_location='cpu')
    tensor_type = first_sample.get('model_type', 'tfpp')
    print(f'  Detected tensor format: {tensor_type}')

    if model_type == 'tfpp' and tensor_type == 'tfpp':
        _run_tfpp_analysis(args, model, config, device, pt_files, output_dir)
    elif model_type == 'plant2' or tensor_type == 'plant2':
        _run_plant2_analysis(args, model, device, pt_files, output_dir)
    else:
        raise ValueError(f"Unsupported model/tensor combination: model={model_type}, tensors={tensor_type}")


def _apply_degradation(tensor, modality, method, strength, device):
    """Apply degradation to a sensor modality tensor.

    Args:
        tensor: Input tensor (B, C, H, W) on device
        modality: 'rgb' or 'lidar'
        method: 'blur', 'noise', 'dropout', or 'zero'
        strength: Float 0.0-1.0 controlling degradation intensity
        device: torch device

    Returns:
        Degraded tensor (same shape, same device)
    """
    if method == 'zero':
        return torch.zeros_like(tensor)

    if method == 'blur' and modality == 'rgb':
        img = tensor[0].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 255).astype(np.float32)
        ksize = int(3 + strength * 48) | 1
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        result = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        return result.to(device)

    if method == 'noise':
        noise_std = strength * tensor.abs().max().item() * 0.5
        noise = torch.randn_like(tensor) * noise_std
        return tensor + noise

    if method == 'dropout' and modality == 'lidar':
        mask = torch.rand_like(tensor) > strength
        return tensor * mask.float()

    raise ValueError(f"Invalid degradation: method={method} for modality={modality}")


def _rgb_to_bgr(rgb_tensor):
    """Convert RGB tensor [0, 255] (C, H, W) to BGR uint8 (H, W, 3)."""
    img = rgb_tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _lidar_to_bgr(lidar_tensor, width):
    """Convert LiDAR BEV tensor (C, H, W) to BGR uint8 (width, width, 3)."""
    lidar_np = lidar_tensor.cpu().numpy()[0]
    gray = (255 - (np.clip(lidar_np, 0, 1) * 255)).astype(np.uint8)
    bgr = np.stack([gray, gray, gray], axis=-1)
    return cv2.resize(bgr, (width, width), interpolation=cv2.INTER_NEAREST)


def _save_input_image(rgb_tensor, lidar_bev_tensor, save_path, step):
    """Save the model input (RGB + LiDAR BEV) as a single image."""
    rgb_bgr = _rgb_to_bgr(rgb_tensor)
    h, w = rgb_bgr.shape[:2]
    lidar_bgr = _lidar_to_bgr(lidar_bev_tensor, w)

    combined = np.concatenate([rgb_bgr, lidar_bgr], axis=0)
    cv2.putText(combined, 'Input RGB', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined, 'Input RGB', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(combined, 'Input LiDAR BEV', (10, h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined, 'Input LiDAR BEV', (10, h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / f'input_{step:04d}.png'), combined)


def _render_comparison(rgb_original, rgb_degraded, lidar_original, lidar_degraded,
                       degrade_modality, degrade_info, save_path, step):
    """Render side-by-side comparison of original vs degraded inputs."""
    rgb_orig_bgr = _rgb_to_bgr(rgb_original)
    rgb_deg_bgr = _rgb_to_bgr(rgb_degraded)
    h, w = rgb_orig_bgr.shape[:2]

    lidar_orig_bgr = _lidar_to_bgr(lidar_original, w)
    lidar_deg_bgr = _lidar_to_bgr(lidar_degraded, w)

    left = np.concatenate([rgb_orig_bgr, lidar_orig_bgr], axis=0)
    right = np.concatenate([rgb_deg_bgr, lidar_deg_bgr], axis=0)

    sep = np.ones((left.shape[0], 4, 3), dtype=np.uint8) * 128
    combined = np.concatenate([left, sep, right], axis=1)

    cv2.putText(combined, 'Original', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined, 'Original', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    x_right = w + 4 + 10
    cv2.putText(combined, f'Degraded: {degrade_info}', (x_right, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined, f'Degraded: {degrade_info}', (x_right, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f'comparison_{step:04d}.png'
    cv2.imwrite(str(save_dir / filename), combined)


def _run_tfpp_analysis(args, model, config, device, pt_files, output_dir):
    """Run XAI analysis for TF++ (LidarCenterNet) tensors."""
    engine = XAIEngine(model, config, device=device)
    visualizer = XAIVisualizer(config)

    aggregate_stats = {method: {head: {'rgb_importance': [], 'lidar_importance': []}
                                 for head in args.output_heads}
                       for method in args.methods}

    degrade_info = None
    if getattr(args, 'degrade', None):
        degrade_info = f'{args.degrade} {args.degrade_method} {args.degrade_strength}'

    print(f'\nRunning TF++ XAI analysis:')
    print(f'  Methods: {args.methods}')
    print(f'  Output heads: {args.output_heads}')
    print(f'  Samples: {len(pt_files)}')
    if degrade_info:
        print(f'  Degradation: {degrade_info}')
    print(f'  Output: {output_dir}\n')

    for pt_file in tqdm(pt_files, desc='XAI Analysis (TF++)'):
        sample = torch.load(pt_file, map_location=device)

        rgb = sample['rgb'].to(device)
        lidar_bev = sample['lidar_bev'].to(device)
        target_point = sample['target_point'].to(device)
        ego_vel = sample['ego_vel'].to(device)
        command = sample['command'].to(device)
        step = sample.get('step', int(pt_file.stem))

        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        if lidar_bev.dim() == 3:
            lidar_bev = lidar_bev.unsqueeze(0)
        if target_point.dim() == 1:
            target_point = target_point.unsqueeze(0)
        if ego_vel.dim() == 1:
            ego_vel = ego_vel.unsqueeze(0)
        if command.dim() == 1:
            command = command.unsqueeze(0)

        if degrade_info:
            rgb_original = rgb.clone()
            lidar_original = lidar_bev.clone()
            if args.degrade == 'rgb':
                rgb = _apply_degradation(rgb, 'rgb', args.degrade_method, args.degrade_strength, device)
            elif args.degrade == 'lidar':
                lidar_bev = _apply_degradation(lidar_bev, 'lidar', args.degrade_method, args.degrade_strength, device)

            sample_dir = output_dir / f'{step:04d}'
            _render_comparison(
                rgb_original[0], rgb[0], lidar_original[0], lidar_bev[0],
                args.degrade, degrade_info, str(sample_dir), step)

        sample_dir = output_dir / f'{step:04d}'
        _save_input_image(rgb[0], lidar_bev[0], str(sample_dir), step)

        for method in args.methods:
            for output_head in args.output_heads:
                try:
                    attributions = engine.compute_attribution(
                        method=method,
                        output_head=output_head,
                        rgb=rgb,
                        lidar_bev=lidar_bev,
                        target_point=target_point,
                        ego_vel=ego_vel,
                        command=command,
                        attributed_modalities=('rgb', 'lidar'),
                    )

                    sample_dir = output_dir / f'{step:04d}'
                    visualizer.render_combined(
                        rgb_tensor=rgb[0],
                        lidar_bev_tensor=lidar_bev[0],
                        attributions={k: v[0] for k, v in attributions.items()},
                        method_name=method,
                        output_head=output_head,
                        save_path=str(sample_dir),
                        step=step,
                        degrade_info=degrade_info,
                    )

                    rgb_imp = attributions['rgb'].abs().sum().item()
                    lidar_imp = attributions['lidar'].abs().sum().item()
                    total = rgb_imp + lidar_imp + 1e-8
                    aggregate_stats[method][output_head]['rgb_importance'].append(rgb_imp / total)
                    aggregate_stats[method][output_head]['lidar_importance'].append(lidar_imp / total)

                except Exception as e:
                    print(f"\n  Warning: {method}/{output_head} failed for step {step}: {e}")
                    continue

        if args.compute_attention:
            try:
                attention = engine.compute_attention(rgb, lidar_bev, target_point, ego_vel, command)
                if 'fusion_attention' in attention:
                    sample_dir = output_dir / f'{step:04d}'
                    visualizer.render_attention_flow(
                        attention['fusion_attention'],
                        image_shape=rgb.shape[2:],
                        lidar_shape=lidar_bev.shape[2:],
                        save_path=str(sample_dir),
                        step=step,
                    )
            except Exception as e:
                print(f"\n  Warning: attention extraction failed for step {step}: {e}")

    degradation_meta = None
    if degrade_info:
        degradation_meta = {
            'modality': args.degrade,
            'method': args.degrade_method,
            'strength': args.degrade_strength,
        }
    _print_and_save_summary(aggregate_stats, args.methods, args.output_heads, output_dir,
                            degradation=degradation_meta)


def _run_plant2_analysis(args, model, device, pt_files, output_dir):
    """Run XAI analysis for PlanT2 tensors (token-level attribution)."""
    from captum.attr import Saliency, IntegratedGradients
    from xai.plant_visualization import PlanTXAIVisualizer

    raw_model = model.model if hasattr(model, 'model') else model
    raw_model.to(device)
    raw_model.train()
    for module in raw_model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                               torch.nn.SyncBatchNorm, torch.nn.LayerNorm)):
            module.eval()

    config = GlobalConfig()
    visualizer = PlanTXAIVisualizer(config)

    plant2_heads = [h for h in args.output_heads if h in ('target_speed', 'checkpoint', 'waypoint')]
    if not plant2_heads:
        plant2_heads = ['waypoint']
        print(f'  Note: Defaulting to output_heads={plant2_heads} for PlanT2')

    unsupported = [m for m in args.methods if m in ('deeplift', 'feature_ablation', 'grad_cam')]
    if unsupported:
        print(f'  Note: Skipping {unsupported} for PlanT2 — these methods manipulate the '
              f'input batch dimension, which is incompatible with the token-to-batch '
              f'remapping via batch_idxs in PlanT2.')
    plant2_methods = [m for m in args.methods if m in ('saliency', 'integrated_gradients')]
    if not plant2_methods:
        plant2_methods = ['saliency']
        print(f'  Note: Defaulting to methods={plant2_methods} for PlanT2')

    aggregate_stats = {method: {head: {'token_importance_mean': [], 'num_active_tokens': []}
                                 for head in plant2_heads}
                       for method in plant2_methods}

    print(f'\nRunning PlanT2 XAI analysis (token-level):')
    print(f'  Methods: {plant2_methods}')
    print(f'  Output heads: {plant2_heads}')
    print(f'  Samples: {len(pt_files)}')
    print(f'  Output: {output_dir}\n')

    class PlanT2ForwardWrapper(torch.nn.Module):
        """Wraps PlanT2 for Captum: continuous features as input, scalar output.

        Key insight: x_objs[:, 0] is the type indicator (discrete category).
        It's used in a boolean mask (x_objs[...,0] == i) which has zero gradient.
        For IG, interpolating it creates non-integer values that match no type.

        Solution: attribute only to continuous features (columns 1-6),
        keeping the type column fixed via additional_forward_args.

        Note: DeepLift and FeatureAblation are not supported because they
        manipulate the input batch dimension, which breaks the token-to-batch
        remapping via batch_idxs inside the PlanT2 model.
        """
        def __init__(self, hflm_model, output_head):
            super().__init__()
            self.hflm = hflm_model
            self.output_head = output_head
            self.batch_template = None

        def forward(self, x_objs_features, x_objs_types):
            x_objs = torch.cat([x_objs_types, x_objs_features], dim=-1)

            batch = dict(self.batch_template)
            batch['x_objs'] = x_objs
            if hasattr(self.hflm, 'input_bev') and self.hflm.input_bev and 'BEV' not in batch:
                batch['BEV'] = torch.zeros(1, 3, 128, 128, device=x_objs.device)
            _, _, pred_plan, _ = self.hflm(batch)
            pred_path, pred_wps, pred_speed = pred_plan

            if self.output_head == 'target_speed' and pred_speed is not None:
                return pred_speed.sum(dim=-1)
            elif self.output_head == 'waypoint' and pred_wps is not None:
                return pred_wps.norm(dim=-1).sum(dim=-1)
            elif self.output_head == 'checkpoint' and pred_path is not None:
                return pred_path.norm(dim=-1).sum(dim=-1)
            elif pred_wps is not None:
                return pred_wps.norm(dim=-1).sum(dim=-1)
            elif pred_path is not None:
                return pred_path.norm(dim=-1).sum(dim=-1)
            else:
                raise ValueError(f"No valid output for head '{self.output_head}'")

    for pt_file in tqdm(pt_files, desc='XAI Analysis (PlanT2)'):
        sample = torch.load(pt_file, map_location=device)
        step = sample.get('step', int(pt_file.stem))

        x_objs = sample['x_objs'].to(device)
        idxs = sample['idxs'].to(device)
        route_original = sample['route_original'].to(device)
        speed_limit = sample['speed_limit'].to(device)

        if route_original.dim() == 2:
            route_original = route_original.unsqueeze(0)
        if speed_limit.dim() == 0:
            speed_limit = speed_limit.unsqueeze(0)
        if idxs.dim() == 1:
            idxs = idxs.unsqueeze(0)

        batch = {
            'x_objs': x_objs,
            'idxs': idxs,
            'route_original': route_original,
            'speed_limit': speed_limit,
            'y_objs': None,
        }
        if 'ego_speed' in sample:
            es = sample['ego_speed'].to(device)
            if es.dim() == 0:
                es = es.unsqueeze(0)
            batch['ego_speed'] = es

        for method in plant2_methods:
            for output_head in plant2_heads:
                try:
                    wrapper = PlanT2ForwardWrapper(raw_model, output_head)
                    wrapper.batch_template = batch

                    # Split x_objs into types (col 0, constant) and features (cols 1-6, attributed)
                    x_objs_types = x_objs[:, 0:1].detach()
                    x_objs_features = x_objs[:, 1:].detach().requires_grad_(True)

                    with torch.enable_grad():
                        if method == 'saliency':
                            attr_method = Saliency(wrapper)
                            attrs = attr_method.attribute(
                                x_objs_features,
                                additional_forward_args=(x_objs_types,))
                        elif method == 'integrated_gradients':
                            attr_method = IntegratedGradients(wrapper)
                            baseline = torch.zeros_like(x_objs_features)
                            attrs = attr_method.attribute(
                                x_objs_features, baselines=baseline,
                                additional_forward_args=(x_objs_types,),
                                n_steps=50)

                    # Compute per-token importance (sum over attributes, skip padding at idx 0)
                    token_importance = attrs.abs().sum(dim=-1)[1:]  # skip padding token
                    active_mask = (x_objs[1:].abs().sum(dim=-1) > 0)
                    token_importance = token_importance * active_mask.float()

                    # Normalize
                    max_imp = token_importance.max()
                    if max_imp > 0:
                        token_importance_norm = token_importance / max_imp
                    else:
                        token_importance_norm = token_importance

                    # Visualize
                    sample_dir = output_dir / f'{step:04d}'
                    sample_dir.mkdir(parents=True, exist_ok=True)

                    visualizer.render_combined(
                        bounding_boxes=x_objs[1:].detach(),
                        token_importance=token_importance_norm.detach(),
                        method_name=method,
                        output_head=output_head,
                        save_path=str(sample_dir),
                        step=step,
                    )

                    # Aggregate
                    imp_np = token_importance.detach().cpu().numpy()
                    active_np = active_mask.cpu().numpy()
                    aggregate_stats[method][output_head]['token_importance_mean'].append(
                        float(imp_np[active_np].mean()) if active_np.any() else 0.0)
                    aggregate_stats[method][output_head]['num_active_tokens'].append(int(active_np.sum()))

                except Exception as e:
                    print(f"\n  Warning: {method}/{output_head} failed for step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    # Summary
    print('\n' + '=' * 60)
    print('AGGREGATE RESULTS (PlanT2 Token-Level)')
    print('=' * 60)

    summary = {}
    for method in plant2_methods:
        summary[method] = {}
        for head in plant2_heads:
            stats = aggregate_stats[method][head]
            if stats['token_importance_mean']:
                imp_mean = np.mean(stats['token_importance_mean'])
                imp_std = np.std(stats['token_importance_mean'])
                tok_mean = np.mean(stats['num_active_tokens'])
                summary[method][head] = {
                    'token_importance_mean': float(imp_mean),
                    'token_importance_std': float(imp_std),
                    'avg_active_tokens': float(tok_mean),
                    'num_samples': len(stats['token_importance_mean']),
                }
                print(f'  {method:25s} / {head:15s}: '
                      f'Importance={imp_mean:.4f} (+/-{imp_std:.4f}), '
                      f'Active tokens={tok_mean:.1f} '
                      f'[n={len(stats["token_importance_mean"])}]')

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nSummary: {summary_path}')
    print(f'Visualizations: {output_dir}/')


def _print_and_save_summary(aggregate_stats, methods, output_heads, output_dir,
                            degradation=None):
    """Print and save aggregate summary for TF++ analysis."""
    print('\n' + '=' * 60)
    print('AGGREGATE RESULTS')
    if degradation:
        print(f'  Degradation: {degradation["modality"]} {degradation["method"]} '
              f'strength={degradation["strength"]}')
    print('=' * 60)

    summary = {}
    if degradation:
        summary['degradation'] = degradation
    for method in methods:
        summary[method] = {}
        for head in output_heads:
            stats = aggregate_stats[method][head]
            if stats['rgb_importance']:
                rgb_mean = np.mean(stats['rgb_importance'])
                rgb_std = np.std(stats['rgb_importance'])
                lidar_mean = np.mean(stats['lidar_importance'])
                lidar_std = np.std(stats['lidar_importance'])
                summary[method][head] = {
                    'rgb_importance_mean': float(rgb_mean),
                    'rgb_importance_std': float(rgb_std),
                    'lidar_importance_mean': float(lidar_mean),
                    'lidar_importance_std': float(lidar_std),
                    'num_samples': len(stats['rgb_importance']),
                }
                print(f'  {method:25s} / {head:15s}: '
                      f'RGB={rgb_mean:.1%} (+/-{rgb_std:.1%}), '
                      f'LiDAR={lidar_mean:.1%} (+/-{lidar_std:.1%}) '
                      f'[n={len(stats["rgb_importance"])}]')

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nSummary: {summary_path}')
    print(f'Visualizations: {output_dir}/')


def main():
    parser = argparse.ArgumentParser(
        description='Offline XAI analysis for fail2drive models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick saliency analysis on 10 samples:
  python -m xai.analysis \\
      --checkpoint ./checkpoints/tfpp \\
      --tensor_dir ./eval_output/route_0/xai_tensors \\
      --output_dir ./xai_results \\
      --methods saliency \\
      --num_samples 10

  # Full analysis with multiple methods:
  python -m xai.analysis \\
      --checkpoint ./checkpoints/tfpp \\
      --tensor_dir ./eval_output/route_0/xai_tensors \\
      --output_dir ./xai_results \\
      --methods saliency integrated_gradients grad_cam deeplift \\
      --output_heads target_speed checkpoint waypoint \\
      --attention
        """)

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model directory (containing config.json + model_*.pth)')
    parser.add_argument('--tensor_dir', type=str, required=True,
                        help='Directory with .pt tensor snapshots from evaluation '
                             '(generated with XAI_ENABLED=1 SAVE_PATH=...)')
    parser.add_argument('--output_dir', type=str, default='./xai_results',
                        help='Output directory for XAI visualizations and statistics')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['saliency', 'integrated_gradients'],
                        choices=['saliency', 'integrated_gradients', 'grad_cam', 'feature_ablation', 'deeplift'],
                        help='XAI methods to compute')
    parser.add_argument('--output_heads', type=str, nargs='+',
                        default=['target_speed', 'checkpoint'],
                        choices=['target_speed', 'checkpoint', 'waypoint', 'bbox', 'semantic'],
                        help='Model output heads to explain')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Max number of samples to analyze (default: all)')
    parser.add_argument('--attention', dest='compute_attention', action='store_true',
                        help='Also extract and visualize cross-modal attention from GPT fusion blocks')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: cuda if available, else cpu)')

    parser.add_argument('--run_name', type=str, default=None,
                        help='Custom run name for output directory (useful for grouping '
                             'degradation variants in the same folder)')

    parser.add_argument('--degrade', type=str, default=None,
                        choices=['rgb', 'lidar'],
                        help='Degrade a sensor modality to test model dependence (TF++ only)')
    parser.add_argument('--degrade_method', type=str, default='blur',
                        choices=['blur', 'noise', 'dropout', 'zero'],
                        help='Degradation method (blur: RGB only, dropout: LiDAR only, '
                             'noise/zero: both)')
    parser.add_argument('--degrade_strength', type=float, default=0.5,
                        help='Degradation strength 0.0-1.0 (default: 0.5)')

    args = parser.parse_args()

    if args.degrade:
        if args.degrade_method == 'blur' and args.degrade != 'rgb':
            parser.error('--degrade_method blur is only valid with --degrade rgb')
        if args.degrade_method == 'dropout' and args.degrade != 'lidar':
            parser.error('--degrade_method dropout is only valid with --degrade lidar')
        if not 0.0 <= args.degrade_strength <= 1.0:
            parser.error('--degrade_strength must be between 0.0 and 1.0')

    run_analysis(args)


if __name__ == '__main__':
    main()