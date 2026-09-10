"""Audit actual teacher sample alignment and BC steady-state bias on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from task_e_geometry import DEFAULT_JOINT_POS
from tools.task_e.il.common import make_model, sha256


def arrays(path):
    chunks = []
    for name in sorted(path.glob('chunk_*.npz')):
        with np.load(name,allow_pickle=False) as data:
            chunks.append({key:data[key].copy() for key in data.files})
    if not chunks:
        raise ValueError(f'No data chunks in {path}')
    return {key:np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}


def error_stats(value):
    return {'mae_rad':float(np.mean(np.abs(value))),
            'p99_abs_rad':float(np.quantile(np.abs(value),.99)),
            'max_abs_rad':float(np.max(np.abs(value)))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--rollout',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--train-reference',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data=arrays(args.dataset)
    meta=json.loads((args.dataset/'metadata.json').read_text())
    result=json.loads((args.rollout/'result.json').read_text())
    with np.load(args.rollout/'telemetry.npz',allow_pickle=False) as telemetry:
        qpos=telemetry['qpos'].copy()
        action=telemetry['action'].copy()
    steps=data['step'].astype(int)
    if np.min(steps)<2 or np.max(steps)>len(qpos) or not np.all(np.diff(steps)>0):
        raise ValueError('Invalid sample step sequence')
    # Evaluator qpos[k] is AFTER control step k+1. Sample step s therefore
    # consumes qpos[s-2] and its teacher action is telemetry.action[s-1].
    pre_q=qpos[steps-2]
    command=action[steps-1]*.5+DEFAULT_JOINT_POS
    delta_from_executed=command[:,:6]-data['q'][:,:6]
    alignment={'pre_action_q_max_abs_error':float(np.max(np.abs(pre_q-data['q']))),
               'teacher_delta_vs_executed_action_max_abs_error':float(np.max(np.abs(delta_from_executed-data['delta']))),
               'label_times_limit_max_abs_error':float(np.max(np.abs(data['label']*data['limit'][:,None]-data['delta'])))}
    source_manifest=json.loads((args.rollout/'source_manifest.json').read_text())
    source_matches={name:source_manifest.get(str(ROOT/name),{}).get('sha256')==digest
                    for name,digest in meta['source_hashes'].items()}
    phase_object=np.asarray([f'{o}/{p}' for o,p in zip(data['object_id'],data['phase'])])
    report={'dataset':str(args.dataset),'rollout':str(args.rollout),'rollout_seed':result['seed'],
            'rollout_score':result['score'],'rollout_steps':result['steps'],
            'samples':len(steps),'first_servo_step':int(steps[0]),'last_servo_step':int(steps[-1]),
            'last_recorded_policy_step':meta['last_policy_step'],
            'finalized_before_sim_shutdown':meta.get('finalized_before_sim_shutdown',False),
            'phase_counts':dict(Counter(data['phase'].tolist())),
            'object_phase_counts':dict(Counter(phase_object.tolist())),
            'all_finite':bool(all(np.isfinite(data[key]).all() for key in ('features','label','delta','q','qdot','target'))),
            'alignment':alignment,'source_snapshots_match':source_matches,
            'saturated_label_fraction':float(np.mean(np.abs(data['label'])>.99)),
            'near_zero_label_fraction':float(np.mean(np.abs(data['label'])<.05)),
            'interpretation':'Offline diagnostic truth comparison only; telemetry truth is not a network input.'}
    if args.train_reference:
        train=arrays(args.train_reference)
        lo=train['features'].reshape(-1,data['features'].shape[-1]).min(axis=0)
        hi=train['features'].reshape(-1,data['features'].shape[-1]).max(axis=0)
        outside=(data['features']<lo-1e-6)|(data['features']>hi+1e-6)
        train_pairs=set(f'{o}/{p}' for o,p in zip(train['object_id'],train['phase']))
        report['distribution_shift']={'feature_range_exceedance_fraction':outside.mean(axis=(0,1)).tolist(),
                                      'unseen_object_phase_pairs':sorted(set(phase_object)-train_pairs),
                                      'note':'Marginal range diagnostic, not a guarantee of in-distribution joint feature combinations.'}
    if args.checkpoint:
        import torch
        torch.set_num_threads(4)
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        model=make_model().eval()
        model.load_state_dict(checkpoint['model_state_dict'])
        x=torch.from_numpy(data['features'].astype(np.float32))
        with torch.inference_mode():
            predicted=model(x).squeeze(-1).numpy()*data['limit'][:,None]
            # Counterfactual zero-error/zero-velocity inputs measure a learned
            # steady-state bias; these are explicitly not observed demonstrations.
            zero=x.clone()
            zero[:,:,0]=0.;zero[:,:,2]=0.
            equilibrium=model(zero).squeeze(-1).numpy()*data['limit'][:,None]
        error=predicted-data['delta']
        report['checkpoint_sha256']=sha256(args.checkpoint)
        report['offline_action_error']=error_stats(error)
        report['counterfactual_equilibrium_bias']=error_stats(equilibrium)
        report['object_phase_action_error']={pair:error_stats(error[phase_object==pair]) for pair in sorted(set(phase_object))}
        active=np.abs(data['delta'])>.002
        report['wrong_direction_fraction_active']=float(np.mean((predicted*data['delta']<0)[active]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
