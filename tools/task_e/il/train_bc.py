"""Train the servo BC network on CPU with whole-episode/seed validation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tools.task_e.il.common import FEATURE_NAMES, INTERFACE_VERSION, make_model, sha256


def load_episodes(paths):
    features, labels, limits = [], [], []
    provenance = []
    for path in paths:
        path = path.expanduser().resolve(strict=True)
        meta = json.loads((path/'metadata.json').read_text())
        if not meta.get('finalized_before_sim_shutdown', False):
            raise ValueError(f'Dataset was not finalized before simulator shutdown: {path}')
        if meta['interface_version'] != INTERFACE_VERSION or meta['feature_names'] != FEATURE_NAMES:
            raise ValueError(f'Incompatible dataset: {path}')
        chunks = sorted(path.glob('chunk_*.npz'))
        if len(chunks) != meta['chunks']:
            raise ValueError(f'Dataset metadata/chunk count differs: {path}')
        count, hashes = 0, {}
        for chunk in chunks:
            with np.load(chunk, allow_pickle=False) as data:
                x = np.asarray(data['features'],dtype=np.float32)
                y = np.asarray(data['label'],dtype=np.float32)
                limit = np.repeat(np.asarray(data['limit'],dtype=np.float32)[:,None],6,axis=1)
                if (x.shape[1:] != (6,len(FEATURE_NAMES)) or y.shape != x.shape[:2]
                        or not np.isfinite(x).all() or not np.isfinite(y).all()
                        or np.max(np.abs(y)) > 1.0001):
                    raise ValueError(f'Invalid training arrays: {chunk}')
                features.append(x.reshape(-1,len(FEATURE_NAMES)))
                labels.append(y.reshape(-1,1))
                limits.append(limit.reshape(-1,1))
                count += len(x)
            hashes[chunk.name] = sha256(chunk)
        if count != meta['samples'] or count < 100:
            raise ValueError(f'Dataset incomplete or too short: {path}, {count} servo frames')
        provenance.append({'directory':str(path), 'episode_id':meta['episode_id'],
                           'seed':meta['rollout_seed'], 'servo_frames':count,
                           'metadata_sha256':sha256(path/'metadata.json'), 'chunk_sha256':hashes})
    return (torch.from_numpy(np.concatenate(features)), torch.from_numpy(np.concatenate(labels)),
            torch.from_numpy(np.concatenate(limits)), provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,nargs='+',required=True)
    parser.add_argument('--validation',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=7)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    x,y,limit,train = load_episodes(args.train)
    vx,vy,vlimit,validation = load_episodes(args.validation)
    if ({p['episode_id'] for p in train} & {p['episode_id'] for p in validation}
            or {p['directory'] for p in train} & {p['directory'] for p in validation}
            or {p['seed'] for p in train} & {p['seed'] for p in validation}):
        raise ValueError('Training and validation must use different complete episodes and seeds')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    started = time.monotonic()
    best_loss, best_state, best_epoch = float('inf'), None, 0
    history = []
    for epoch in range(1,args.epochs+1):
        model.train()
        order = torch.randperm(len(x))
        losses = []
        for indices in order.split(args.batch_size):
            predicted = model(x[indices])
            # Small corrections determine waypoint settling; emphasize those
            # observed labels without generating synthetic teacher samples.
            weight = 1.+4.*(y[indices].abs() < .35)
            loss = ((predicted-y[indices]).square()*weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            predicted = model(vx)
            validation_loss = float(((predicted-vy).square()*(1.+4.*(vy.abs()<.35))).mean())
            error_rad = (predicted-vy)*vlimit
            validation_mae_rad = float(error_rad.abs().mean())
        if validation_loss < best_loss:
            best_loss, best_state, best_epoch = validation_loss,copy.deepcopy(model.state_dict()),epoch
        row={'epoch':epoch,'training_weighted_mse':float(np.mean(losses)),
             'validation_weighted_mse':validation_loss,'validation_joint_delta_mae_rad':validation_mae_rad}
        history.append(row)
        if epoch%20==0 or epoch==1:
            print('TASK_E_BC_TRAIN '+json.dumps(row),flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        error = ((model(vx)-vy)*vlimit).numpy().reshape(-1)
        zero_error = (vy*vlimit).numpy().reshape(-1)
    checkpoint = {'interface_version':INTERFACE_VERSION,'feature_names':FEATURE_NAMES,
                  'model_state_dict':best_state,'training_seed':args.seed,'best_epoch':best_epoch,
                  'train_episode_ids':[p['episode_id'] for p in train],
                  'validation_episode_ids':[p['episode_id'] for p in validation],
                  'scope':'Behavior cloning of six-joint servo only; hand-coded perception/IK/FSM retained'}
    checkpoint_path = args.output/'servo_bc.pt'
    torch.save(checkpoint,checkpoint_path)
    report={'train':train,'validation':validation,'device':'cpu','epochs':args.epochs,
            'best_epoch':best_epoch,'training_joint_samples':len(x),'validation_joint_samples':len(vx),
            'wall_seconds':time.monotonic()-started,'checkpoint_sha256':sha256(checkpoint_path),
            'validation_delta_mae_rad':float(np.mean(np.abs(error))),
            'validation_delta_rmse_rad':float(np.sqrt(np.mean(error**2))),
            'validation_delta_p99_abs_rad':float(np.quantile(np.abs(error),.99)),
            'zero_action_baseline_mae_rad':float(np.mean(np.abs(zero_error))),
            'closed_loop_evaluation_required':True,'full_score_claim':False,
            'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha256(__file__),
                             'tools/task_e/il/common.py':sha256(ROOT/'tools/task_e/il/common.py')},
            'history':history}
    (args.output/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('TASK_E_BC_RESULT '+json.dumps({k:v for k,v in report.items() if k not in ('history','train','validation')}),flush=True)


if __name__=='__main__':
    main()
