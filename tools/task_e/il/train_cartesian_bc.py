"""Train the Cartesian-goal residual servo network on CPU with whole-seed validation.

Features and labels are derived offline from complete existing demonstration
chunks: the stored joint-space waypoint becomes a Cartesian goal, the analytic
DLS step is recomputed, and the label is the teacher joint delta minus that
analytic step. The teacher action is used only as a label, never as an input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tools.task_e.il.cartesian import (CARTESIAN_FEATURE_NAMES, CARTESIAN_INTERFACE_VERSION,
                                       command_from_residual, derive_sample,
                                       make_cartesian_model, residual_scale_from_labels)
from tools.task_e.il.common import FEATURE_NAMES, INTERFACE_VERSION, sha256

STORED_KEYS = ('q', 'qdot', 'target', 'delta', 'limit', 'phase', 'object_id', 'step', 'score')


def load_dataset(paths):
    """Load complete finalized demonstrations, preserving raw file hashes."""
    episodes, provenance = [], []
    for path in paths:
        path = path.expanduser().resolve(strict=True)
        meta = json.loads((path/'metadata.json').read_text())
        if not meta.get('finalized_before_sim_shutdown', False):
            raise ValueError(f'Dataset was not finalized before simulator shutdown: {path}')
        if meta['interface_version'] != INTERFACE_VERSION or meta['feature_names'] != FEATURE_NAMES:
            raise ValueError(f'Incompatible collected dataset: {path}')
        chunks = sorted(path.glob('chunk_*.npz'))
        if len(chunks) != meta['chunks']:
            raise ValueError(f'Dataset metadata/chunk count differs: {path}')
        parts, hashes, count = [], {}, 0
        for chunk in chunks:
            with np.load(chunk, allow_pickle=False) as data:
                if any(key not in data.files for key in STORED_KEYS):
                    raise ValueError(f'Dataset chunk is missing stored servo fields: {chunk}')
                part = {key: data[key].copy() for key in STORED_KEYS}
            frames = len(part['step'])
            if (part['q'].shape[1] < 6 or part['qdot'].shape[1] != 6 or part['target'].shape[1] != 6
                    or part['delta'].shape != (frames, 6) or part['limit'].shape != (frames,)
                    or not all(np.isfinite(part[key]).all()
                               for key in ('q', 'qdot', 'target', 'delta', 'limit'))):
                raise ValueError(f'Invalid stored servo frames: {chunk}')
            if np.max(np.abs(part['delta']) - part['limit'][:, None]) > 1e-6:
                raise ValueError(f'Stored teacher delta exceeds its declared limit: {chunk}')
            parts.append(part)
            hashes[chunk.name] = sha256(chunk)
            count += frames
        if count != meta['samples'] or count < 100:
            raise ValueError(f'Dataset incomplete or too short: {path}, {count} servo frames')
        episode = {key: np.concatenate([part[key] for part in parts]) for key in STORED_KEYS}
        episode['episode_id'] = np.full(count, meta['episode_id'])
        episodes.append(episode)
        provenance.append({'directory': str(path), 'episode_id': meta['episode_id'],
                           'seed': meta['rollout_seed'], 'servo_frames': count,
                           'last_observed_score': meta.get('last_observed_score'),
                           'metadata_sha256': sha256(path/'metadata.json'),
                           'chunk_sha256': hashes,
                           'collector_source_sha256': meta.get('source_hashes', {})})
    return ({key: np.concatenate([episode[key] for episode in episodes])
             for key in episodes[0]}, provenance)


def derive(data):
    """Recompute Cartesian features, DLS steps and residual labels frame by frame."""
    frames = len(data['step'])
    features = np.empty((frames, len(CARTESIAN_FEATURE_NAMES)), dtype=np.float32)
    labels = np.empty((frames, 6), dtype=np.float32)
    analytic = np.empty((frames, 6), dtype=np.float32)
    for i in range(frames):
        features[i], labels[i], analytic[i] = derive_sample(
            data['q'][i], data['qdot'][i], data['target'][i], data['delta'][i],
            float(data['limit'][i]), str(data['phase'][i]), int(data['object_id'][i]))
    if not (np.isfinite(features).all() and np.isfinite(labels).all()):
        raise ValueError('Non-finite derived features or residual labels')
    return features, labels, analytic


def balanced_weights(data):
    """Equal total weight per phase/object group, so long phases cannot dominate."""
    groups = np.asarray([f'{int(o)}/{p}' for o, p in zip(data['object_id'], data['phase'])])
    counts = Counter(groups.tolist())
    weights = np.asarray([1./counts[group] for group in groups], dtype=np.float32)
    return weights/float(weights.mean()), groups, dict(sorted(counts.items()))


def error_stats(predicted_delta, teacher_delta):
    error = (np.asarray(predicted_delta) - np.asarray(teacher_delta)).reshape(-1)
    return {'frames': int(len(np.asarray(teacher_delta))),
            'joint_delta_mae_rad': float(np.mean(np.abs(error))),
            'joint_delta_rmse_rad': float(np.sqrt(np.mean(error**2))),
            'joint_delta_p99_abs_rad': float(np.quantile(np.abs(error), .99)),
            'joint_delta_max_abs_rad': float(np.max(np.abs(error)))}


def commanded_deltas(data, analytic, residual):
    """Offline replay of the deployed guard chain, including clamp counting."""
    frames = len(analytic)
    deltas = np.empty((frames, 6))
    rate_clamped = joint_clamped = 0
    for i in range(frames):
        q = np.asarray(data['q'][i], dtype=float)[:6]
        command, rate, joint = command_from_residual(q, analytic[i], residual[i],
                                                     float(data['limit'][i]))
        deltas[i] = command - q
        rate_clamped += int(rate)
        joint_clamped += int(joint)
    return deltas, {'rate_limit_clamped_frames': rate_clamped,
                    'joint_limit_clamped_frames': joint_clamped}


def phase_breakdown(data, model_delta, dls_delta):
    result = {}
    for phase in sorted(set(data['phase'].tolist())):
        mask = data['phase'] == phase
        result[str(phase)] = {'learned': error_stats(model_delta[mask], data['delta'][mask]),
                              'dls_only': error_stats(dls_delta[mask], data['delta'][mask])}
    return result


def range_shift(train_x, validation_x):
    """Marginal input-range diagnostic between the two disjoint seeds."""
    low, high = train_x.min(axis=0), train_x.max(axis=0)
    mean, spread = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    outside = ((validation_x < low - 1e-6) | (validation_x > high + 1e-6)).mean(axis=0)
    shift = np.abs(validation_x.mean(axis=0) - mean)/spread
    order = np.argsort(-outside)[:8]
    return {'max_feature_range_exceedance_fraction': float(outside.max()),
            'mean_feature_range_exceedance_fraction': float(outside.mean()),
            'max_standardized_mean_shift': float(shift.max()),
            'worst_features': {CARTESIAN_FEATURE_NAMES[i]:
                               {'validation_outside_train_range_fraction': float(outside[i]),
                                'standardized_mean_shift': float(shift[i]),
                                'train_min': float(low[i]), 'train_max': float(high[i]),
                                'validation_min': float(validation_x[:, i].min()),
                                'validation_max': float(validation_x[:, i].max())}
                               for i in order},
            'per_feature_range_exceedance_fraction': dict(zip(CARTESIAN_FEATURE_NAMES,
                                                              outside.tolist())),
            'note': 'Marginal ranges only; not a guarantee of in-distribution joint inputs.'}


def coverage_report(train_labels, validation_labels, scale, quantile):
    within_train = np.abs(train_labels) <= scale
    return {'residual_scale_rad': scale.tolist(), 'scale_quantile': float(quantile),
            'train_within_scale_fraction': float(within_train.mean()),
            'train_per_joint_within_scale_fraction': within_train.mean(axis=0).tolist(),
            'validation_within_scale_fraction': float(np.mean(np.abs(validation_labels) <= scale)),
            'train_label_mae_rad': float(np.mean(np.abs(train_labels))),
            'train_label_p99_abs_rad': float(np.quantile(np.abs(train_labels), .99)),
            'train_label_max_abs_rad': float(np.max(np.abs(train_labels))),
            'validation_label_max_abs_rad': float(np.max(np.abs(validation_labels)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, nargs='+', required=True)
    parser.add_argument('--validation', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--residual-quantile', type=float, default=.995)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    if not .5 <= args.residual_quantile <= 1.:
        parser.error('residual-quantile must lie in [0.5, 1.0]')

    train_data, train_provenance = load_dataset(args.train)
    validation_data, validation_provenance = load_dataset(args.validation)
    if ({p['episode_id'] for p in train_provenance} & {p['episode_id'] for p in validation_provenance}
            or {p['directory'] for p in train_provenance} & {p['directory'] for p in validation_provenance}
            or {p['seed'] for p in train_provenance} & {p['seed'] for p in validation_provenance}):
        raise ValueError('Training and validation must use different complete episodes and seeds')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    started = time.monotonic()
    train_x, train_y, train_analytic = derive(train_data)
    validation_x, validation_y, validation_analytic = derive(validation_data)
    scale = residual_scale_from_labels(train_y, args.residual_quantile)
    weights, _, train_groups = balanced_weights(train_data)
    validation_weights, _, validation_groups = balanced_weights(validation_data)
    x, y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    vx, vy = torch.from_numpy(validation_x), torch.from_numpy(validation_y)
    w = torch.from_numpy(weights).unsqueeze(1)
    vw = torch.from_numpy(validation_weights).unsqueeze(1)
    scale_tensor = torch.from_numpy(scale)

    model = make_cartesian_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    best_loss, best_state, best_epoch = float('inf'), copy.deepcopy(model.state_dict()), 0
    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        order = torch.randperm(len(x))
        losses = []
        for indices in order.split(args.batch_size):
            # Residual units: the loss is scale-free per joint, and equals the
            # DLS-only loss at initialization because the output layer is zero.
            predicted = model(x[indices])*scale_tensor
            loss = (w[indices]*((predicted-y[indices])/scale_tensor).square()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            predicted = model(vx)*scale_tensor
            validation_loss = float((vw*((predicted-vy)/scale_tensor).square()).mean())
            validation_mae = float((predicted-vy).abs().mean())
        if validation_loss < best_loss:
            best_loss, best_state, best_epoch = validation_loss, copy.deepcopy(model.state_dict()), epoch
        row = {'epoch': epoch, 'training_weighted_residual_mse': float(np.mean(losses)),
               'validation_weighted_residual_mse': validation_loss,
               'validation_residual_mae_rad': validation_mae}
        history.append(row)
        if epoch % 20 == 0 or epoch == 1:
            print('TASK_E_CARTESIAN_TRAIN '+json.dumps(row), flush=True)

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        residual = (model(vx)*scale_tensor).numpy().astype(float)
    zero_residual = np.zeros_like(residual)
    model_delta, model_clamps = commanded_deltas(validation_data, validation_analytic, residual)
    dls_delta, dls_clamps = commanded_deltas(validation_data, validation_analytic, zero_residual)
    changed = np.max(np.abs(model_delta-dls_delta), axis=1)

    checkpoint = {'cartesian_interface_version': CARTESIAN_INTERFACE_VERSION,
                  'dataset_interface_version': INTERFACE_VERSION,
                  'feature_names': list(CARTESIAN_FEATURE_NAMES),
                  'residual_scale': scale.astype(np.float32).tolist(),
                  'model_state_dict': {k: v.cpu() for k, v in best_state.items()},
                  'training_seed': args.seed, 'best_epoch': best_epoch,
                  'train_episode_ids': [p['episode_id'] for p in train_provenance],
                  'validation_episode_ids': [p['episode_id'] for p in validation_provenance],
                  'analytic_baseline': 'task_e_geometry.differential_ik on an FK Cartesian goal',
                  'scope': ('Learned bounded joint-delta residual over analytic DLS; RGBD perception, '
                            'IK path, waypoint acceptance, state machine and gripper remain hand-coded.')}
    checkpoint_path = args.output/'cartesian_bc.pt'
    torch.save(checkpoint, checkpoint_path)

    report = {'experiment': 'Cartesian-goal residual BC of the six-joint arm servo',
              'device': 'cpu', 'epochs': args.epochs, 'best_epoch': best_epoch,
              'wall_seconds': time.monotonic()-started,
              'train': train_provenance, 'validation': validation_provenance,
              'train_frames': int(len(train_x)), 'validation_frames': int(len(validation_x)),
              'train_phase_object_counts': train_groups,
              'validation_phase_object_counts': validation_groups,
              'residual_coverage': coverage_report(train_y, validation_y, scale, args.residual_quantile),
              'validation_commanded_delta_error': {
                  'learned': error_stats(model_delta, validation_data['delta']),
                  'dls_only': error_stats(dls_delta, validation_data['delta'])},
              'validation_commanded_delta_error_by_phase': phase_breakdown(
                  validation_data, model_delta, dls_delta),
              'validation_residual_error': {
                  'learned': error_stats(residual, validation_y),
                  'dls_only': error_stats(zero_residual, validation_y)},
              'physical_guard_clamps': {'learned': model_clamps, 'dls_only': dls_clamps},
              'learned_contribution': {
                  'mean_abs_analytic_delta_rad': float(np.mean(np.abs(validation_analytic))),
                  'mean_abs_residual_rad': float(np.mean(np.abs(residual))),
                  'frames_residual_changed_command': int(np.sum(changed > 1e-9)),
                  'fraction_residual_changed_command': float(np.mean(changed > 1e-9)),
                  'max_command_change_from_residual_rad': float(changed.max())},
              'input_range_shift': range_shift(train_x, validation_x),
              'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha256(checkpoint_path),
              'source_sha256': {name: sha256(ROOT/name) for name in
                                ('tools/task_e/il/cartesian.py', 'tools/task_e/il/train_cartesian_bc.py',
                                 'tools/task_e/il/common.py', 'task_e_geometry.py')},
              'closed_loop_evaluation_required': True, 'full_score_claim': False,
              'score_claim': 'None. Offline residual error is not a task score; use the evaluator result.json.',
              'history': history}
    (args.output/'training_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('TASK_E_CARTESIAN_RESULT '+json.dumps(
        {k: v for k, v in report.items()
         if k not in ('history', 'train', 'validation', 'input_range_shift',
                      'train_phase_object_counts', 'validation_phase_object_counts')}), flush=True)


if __name__ == '__main__':
    main()
