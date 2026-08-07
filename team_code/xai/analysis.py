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


def load_model(checkpoint_path, device):
    """Load a LidarCenterNet model from a checkpoint directory.

    The directory must contain config.json and at least one model_*.pth file.
    """
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

    print(f"Loading model: {model_files[0]}")
    state_dict = torch.load(os.path.join(checkpoint_path, model_files[0]), map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, config


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
    model, config = load_model(args.checkpoint, device)

    engine = XAIEngine(model, config, device=device)
    visualizer = XAIVisualizer(config)

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = load_tensor_samples(args.tensor_dir, args.num_samples)

    aggregate_stats = {method: {head: {'rgb_importance': [], 'lidar_importance': []}
                                 for head in args.output_heads}
                       for method in args.methods}

    print(f'\nRunning XAI analysis:')
    print(f'  Methods: {args.methods}')
    print(f'  Output heads: {args.output_heads}')
    print(f'  Samples: {len(pt_files)}')
    print(f'  Output: {output_dir}\n')

    for pt_file in tqdm(pt_files, desc='XAI Analysis'):
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

    print('\n' + '=' * 60)
    print('AGGREGATE RESULTS')
    print('=' * 60)

    summary = {}
    for method in args.methods:
        summary[method] = {}
        for head in args.output_heads:
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
      --methods saliency integrated_gradients grad_cam \\
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
                        choices=['saliency', 'integrated_gradients', 'grad_cam', 'feature_ablation'],
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

    args = parser.parse_args()
    run_analysis(args)


if __name__ == '__main__':
    main()