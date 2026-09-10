"""Independent input-boundary and equilibrium audit for a trained Cartesian model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from task_e_geometry import differential_ik, fk
from tools.task_e.il.common import sha256
from tools.task_e.il.cartesian import (derive_sample,cartesian_features,analytic_step,
                                      make_cartesian_model,command_from_residual)
from tools.task_e.il.train_cartesian_bc import load_dataset


def stats(x):
    return {'mean_abs_rad':float(np.mean(np.abs(x))),
            'p99_abs_rad':float(np.quantile(np.abs(x),.99)),
            'max_abs_rad':float(np.max(np.abs(x)))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4)
    data,provenance=load_dataset([args.dataset])
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=make_cartesian_model().eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    scale=np.asarray(checkpoint['residual_scale'])
    feature_label_difference=feature_equivalent_goal_difference=dls_difference=zero_dls_difference=0.
    features=[];equilibrium_features=[];dls=[];baseline=[]
    # Check every recorded frame, not random frame train/validation partitions.
    for i in range(len(data['q'])):
        q,qdot,target,delta=(data[key][i] for key in ('q','qdot','target','delta'))
        limit,phase,obj=float(data['limit'][i]),str(data['phase'][i]),int(data['object_id'][i])
        x,label,analytic=derive_sample(q,qdot,target,delta,limit,phase,obj)
        alternate_x,_,_=derive_sample(q,qdot,target,delta+.123,limit,phase,obj)
        feature_label_difference=max(feature_label_difference,float(np.max(np.abs(x-alternate_x))))
        # FK is periodic: an equivalent pose encoded with a 2pi joint offset
        # must not expose the hidden joint-space target to the network. This
        # counterfactual tests the feature boundary, not a legal motion target.
        equivalent=target.copy();equivalent[5]+=2*np.pi
        equivalent_x,_,_=derive_sample(q,qdot,equivalent,delta,limit,phase,obj)
        feature_equivalent_goal_difference=max(feature_equivalent_goal_difference,float(np.max(np.abs(x-equivalent_x))))
        goal=fk(target)
        independent=differential_ik(q[:6],goal[:3,3],goal[:3,:3],max_joint_delta=limit)-q[:6]
        dls_difference=max(dls_difference,float(np.max(np.abs(analytic-independent))))
        held=fk(q)
        zero_dls=analytic_step(q,held[:3,3],held[:3,:3],limit)
        zero_dls_difference=max(zero_dls_difference,float(np.max(np.abs(zero_dls))))
        equilibrium_features.append(cartesian_features(q,np.zeros(6),held[:3,3],held[:3,:3],
                                                       zero_dls,limit,phase,obj))
        features.append(x);dls.append(analytic)
        baseline.append(command_from_residual(q,analytic,np.zeros(6),limit)[0]-q[:6])
    with torch.inference_mode():
        residual=model(torch.tensor(np.asarray(features),dtype=torch.float32)).numpy()*scale
        equilibrium=model(torch.tensor(np.asarray(equilibrium_features),dtype=torch.float32)).numpy()*scale
    baseline=np.asarray(baseline)
    commands=np.asarray([command_from_residual(q,a,r,float(l))[0]-q[:6]
                         for q,a,r,l in zip(data['q'],dls,residual,data['limit'])])
    result={'dataset_provenance':provenance,'checkpoint_sha256':sha256(args.checkpoint),
            'audited_frames':len(features),'feature_count':len(features[0]),
            'teacher_label_perturbation_feature_difference':feature_label_difference,
            'equivalent_Cartesian_goal_feature_difference':feature_equivalent_goal_difference,
            'independent_DLS_difference_rad':dls_difference,
            'zero_goal_DLS_max_abs_rad':zero_dls_difference,
            'counterfactual_zero_goal_zero_velocity_learned_residual_bias':stats(equilibrium),
            'learned_teacher_error':stats(commands-data['delta']),
            'dls_teacher_error':stats(baseline-data['delta']),
            'learned_residual_magnitude':stats(residual),
            'phase_equilibrium_bias':{str(phase):stats(equilibrium[data['phase']==phase])
                                      for phase in np.unique(data['phase'])},
            'scope':'CPU audit only. Perturbations test label/target leakage and steady-state bias, not task completion.',
            'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha256(__file__),
                             'tools/task_e/il/cartesian.py':sha256(ROOT/'tools/task_e/il/cartesian.py')}}
    assert feature_label_difference==0 and feature_equivalent_goal_difference<1e-5 and dls_difference<1e-10
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='dataset_provenance'},indent=2))


if __name__=='__main__':main()
