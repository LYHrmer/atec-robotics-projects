"""Verify that recorded learned servo outputs match executed simulator actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from task_e_geometry import DEFAULT_JOINT_POS
from tools.task_e.il.common import sha256
from tools.task_e.il.cartesian import make_cartesian_model,command_from_residual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout',type=Path,required=True)
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=json.loads((args.rollout/'result.json').read_text())
    metrics=json.loads(args.metrics.read_text())
    if not metrics['finalized_before_sim_shutdown']:
        raise ValueError('Cannot audit an unfinished metrics file')
    with np.load(args.metrics.with_suffix('.servo.npz'),allow_pickle=False) as data:
        trace={key:data[key].copy() for key in data.files}
    with np.load(args.rollout/'telemetry.npz',allow_pickle=False) as data:
        actions=data['action'].copy()
    steps=trace['step'].astype(int)
    if not np.all(np.diff(steps)>0) or steps[-1]>len(actions):
        raise ValueError('Invalid servo-to-control-step alignment')
    checkpoint_sha=sha256(args.checkpoint)
    if checkpoint_sha!=metrics['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    torch.set_num_threads(4)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=make_cartesian_model().eval();model.load_state_dict(checkpoint['model_state_dict'])
    with torch.inference_mode():
        recomputed=model(torch.from_numpy(trace['features'].astype(np.float32))).numpy()*np.asarray(checkpoint['residual_scale'])
    is_dls=metrics['mode']=='dls_only_ablation'
    if is_dls:recomputed=np.zeros_like(recomputed)
    reconstructed=np.asarray([command_from_residual(q,a,r,float(l))[0]-q[:6]
                              for q,a,r,l in zip(trace['q'],trace['analytic'],recomputed,trace['limit'])])
    executed=actions[steps-1,:6]*.5+DEFAULT_JOINT_POS[:6]-trace['q'][:,:6]
    # Raw trace is captured before rule-contact-stop overrides. This audit
    # can make an exact all-frame execution claim only when none occurred.
    contact_stops=int(metrics['rule_contact_stop_calls'])
    residual_difference=float(np.max(np.abs(recomputed-trace['residual'])))
    command_difference=float(np.max(np.abs(reconstructed-trace['command_delta'])))
    execution_difference=float(np.max(np.abs(executed-trace['command_delta'])))
    report={'score':result['score'],'full_score':result['full_score'],'seed':result['seed'],
            'steps':result['steps'],'sim_seconds':result['sim_seconds'],'stop_reason':result['stop_reason'],
            'mode':metrics['mode'],'checkpoint_sha256':checkpoint_sha,
            'servo_frames':len(steps),'first_servo_step':int(steps[0]),'last_servo_step':int(steps[-1]),
            'teacher_action_fallback_calls':metrics['teacher_action_fallback_calls'],
            'rule_contact_stop_calls':contact_stops,
            'learned_fraction_all_calls':metrics['learned_fraction_all_calls'],
            'learned_fraction_moving_calls':metrics['learned_fraction_moving_calls'],
            'recomputed_residual_max_abs_error_rad':residual_difference,
            'recomputed_command_max_abs_error_rad':command_difference,
            'recorded_command_vs_executed_action_max_abs_error_rad':execution_difference,
            'all_servo_commands_execution_verified':contact_stops==0 and execution_difference<1e-6,
            'scope':'Offline execution audit; actual task score comes from unchanged evaluator result.json.',
            'files_sha256':{'result.json':sha256(args.rollout/'result.json'),
                            'telemetry.npz':sha256(args.rollout/'telemetry.npz'),
                            'metrics.json':sha256(args.metrics),
                            'servo_trace.npz':sha256(args.metrics.with_suffix('.servo.npz')),
                            'audit_script':sha256(__file__)}}
    assert residual_difference<1e-6 and command_difference<1e-6
    if contact_stops==0:assert execution_difference<1e-6
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
