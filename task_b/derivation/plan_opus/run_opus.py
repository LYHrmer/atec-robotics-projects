import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import argparse

root = Path('/home/lybm/ATEC_Experiments_20260910/task_b_plan_opus')
parser = argparse.ArgumentParser()
parser.add_argument('--task', choices=('plan', 'stationary_gate', 'stationary_gate_resume', 'brake_wheel_hold', 'bounded_visual_reach', 'reach_settle', 'p2_lowering', 'multi_reach', 'grasp_probe', 'grasp_refine', 'grasp_lower03', 'grasp_force75', 'public_odometry', 'contact_grasp'), default='plan')
parser.add_argument('--resume')
args = parser.parse_args()
task = args.task
prompt = root / {'plan':'prompt.md', 'stationary_gate':'stationary_gate_prompt.md',
                 'stationary_gate_resume':'stationary_gate_resume_prompt.md',
                 'brake_wheel_hold':'brake_wheel_hold_prompt.md',
                 'bounded_visual_reach':'bounded_visual_reach_prompt.md',
                 'reach_settle':'reach_settle_prompt.md',
                 'p2_lowering':'p2_lowering_prompt.md',
                 'multi_reach':'multi_reach_prompt.md',
                 'grasp_probe':'grasp_probe_prompt.md',
                 'grasp_refine':'grasp_refine_prompt.md',
                 'grasp_lower03':'grasp_lower03_prompt.md',
                 'grasp_force75':'grasp_force75_prompt.md',
                 'public_odometry':'public_odometry_prompt.md',
                 'contact_grasp':'contact_grasp_prompt.md'}[task]
allowed = 'Read,Glob,Grep' if task == 'plan' else 'Read,Glob,Grep,Write,Edit'
session = args.resume or str(uuid.uuid4())
record_id = session + ('_retry_' + uuid.uuid4().hex[:8] if args.resume else '')
command = [
    '/home/lybm/.npm-global/bin/claude', '--print', '--model', 'opus',
    '--effort', 'low' if task in ('stationary_gate_resume', 'brake_wheel_hold', 'bounded_visual_reach', 'reach_settle', 'p2_lowering', 'multi_reach', 'grasp_probe', 'grasp_refine', 'grasp_lower03', 'grasp_force75', 'public_odometry', 'contact_grasp') else 'high', '--output-format', 'json',
    '--resume' if args.resume else '--session-id', session,
    '--permission-mode', 'dontAsk', '--tools', allowed,
    '--allowedTools', allowed, '--strict-mcp-config',
    '--disable-slash-commands', '--add-dir',
    '/home/lybm/ATEC_Experiments_20260910',
]
record = {
    'provider': 'actual Claude CLI with --model opus',
    'session_id': session,
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'state': 'starting', 'command': command,
    'scope': ('read-only Task B planning' if task == 'plan' else
              'write only task_b/contact_grasp.py per Astra contract; guard owns tests, root owns evaluator; no GPU, Git or memory writes' if task == 'contact_grasp' else
              'write only task_b/public_odometry.py per Astra contract; root owns tests and integration; no GPU, Git or memory writes' if task == 'public_odometry' else
              'write only task_b/grasp_probe.py per Astra contract; independent agent owns tests, root owns evaluator; no GPU, Git or memory writes' if task in ('grasp_probe', 'grasp_refine', 'grasp_lower03', 'grasp_force75') else
              'write only task_b/multi_reach.py per Astra contract; independent agent owns tests, root owns evaluator; no GPU, Git or memory writes' if task == 'multi_reach' else
              'write only first_reach.py for bounded P2 lowering; independent agent owns tests; no GPU, Git or memory writes' if task == 'p2_lowering' else
              'write only first_reach.py and audit_first_reach.py for finite settle; no GPU, Git or memory writes' if task == 'reach_settle' else
              'write only first_reach.py, visual_approach.py, audit_first_reach.py per Astra bounded reach contract; no GPU or other edits' if task == 'bounded_visual_reach' else
              'write only brake_wheel_hold.py and audit_brake_wheel_hold.py; no GPU, no other edits' if task == 'brake_wheel_hold' else
              'write only stationary_target_gate.py and audit_stationary_target_gate.py; no GPU, no other code edits'),
    'task': task,
    'prompt_sha256': hashlib.sha256(prompt.read_bytes()).hexdigest(),
}
audit_path = root / ('invocation_' + record_id + '.json')
result_path = root / ('result_' + record_id + '.json')
stderr_path = root / ('stderr_' + record_id + '.txt')
with prompt.open('rb') as inp, result_path.open('wb') as out, stderr_path.open('wb') as err:
    proc = subprocess.Popen(command, cwd='/home/lybm/ATEC_Robotics_Projects_20260910',
                            stdin=inp, stdout=out, stderr=err)
    record.update(state='running', pid=proc.pid, result_file=str(result_path), stderr_file=str(stderr_path))
    audit_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps({'state': 'running', 'pid': proc.pid, 'session_id': session, 'audit': str(audit_path)}), flush=True)
    code = proc.wait()
record.update(state='process_exited', exit_code=code,
              ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
try:
    payload = json.loads(result_path.read_text())
    record['result_type'] = payload.get('type')
    record['result_subtype'] = payload.get('subtype')
    record['is_error'] = payload.get('is_error')
    record['num_turns'] = payload.get('num_turns')
    record['model_usage'] = payload.get('modelUsage')
    (root / ('answer_' + record_id + '.md')).write_text(payload.get('result', ''))
except Exception as error:
    record['parse_error'] = str(error)
audit_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
print(json.dumps({'state': record['state'], 'exit_code': code, 'session_id': session, 'is_error': record.get('is_error')}), flush=True)
sys.exit(code)
