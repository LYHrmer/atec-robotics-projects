"""Execute two verbatim AST-extracted production methods on isolated CPU objects.

Avoid importing policy dependencies or constructing any simulator. No code changes.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

source=Path('/home/lybm/ATEC_Robotics_Projects_20260910/task_b/payload_motion.py')
tree=ast.parse(source.read_text())
policy=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='PayloadMotionPolicy')
functions=[x for x in policy.body if isinstance(x,ast.FunctionDef) and x.name in ['_hold_tick','_pause']]
constants=['FINAL_HOLD_DEADLINE_S','GOAL_ERROR_RAD','FINAL_HOLD_S','PAUSE_LINEAR_M_S','PAUSE_ANGULAR_RAD_S',
           'PAUSE_EPISODE_S','PAUSE_TOTAL_S','PAUSE_QUIET_S']
assignments=[]
for x in tree.body:
    if isinstance(x,ast.Assign) and any(n.id in constants for t in x.targets for n in ast.walk(t) if isinstance(n,ast.Name)):
        assignments.append(x)
module=ast.Module(body=assignments+functions,type_ignores=[])
env={}
exec(compile(module,str(source),'exec'),env)
class Mock:
    pass
p=Mock();p.calls=120;p.phase_start=0;p.dt=.02;p.quiet_tick=True;p.quiet_ready=False;p.hold_quiet_calls=99
class Tracker:
    done_reason=None
    def update(self,*a,**k):return np.zeros(6)
    def goal_error(self,q):return 0.
p.tracker=Tracker();p.arm_qdot=np.zeros(6);p._stop=lambda reason:reason;p._action=lambda:None;p._record=lambda:None
hold_result=env['_hold_tick'](p,np.zeros(8))
p.pause_start=None;p.pause_calls=0;p.pause_quiet_calls=0;p.pause_episodes=0
p.motion={'linear_norm':.1,'angular_norm':0.};p.hold_quiet_calls=50
pause_result=env['_pause'](p)
x={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
   'verbatim_production_methods_ast_extracted_without_importing_dependencies':True,
   'hold_without_position_window_result':hold_result,'pause_result':pause_result,
   'hold_quiet_calls_after_pause_start':p.hold_quiet_calls,
   'velocity_lowpass_transfer':{str(tau):{str(f):float(abs((1-np.exp(-.02/tau))/(1-np.exp(-.02/tau)*np.exp(-2j*np.pi*f*.02)))) for f in [1.67,25]} for tau in [.04,.1]}}
out=source.parent.parent.parent/'ATEC_Experiments_20260910/task_b_astra_review_20260914/payload_hold_counterexamples_2006.json'
out.write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps(x,indent=2))
