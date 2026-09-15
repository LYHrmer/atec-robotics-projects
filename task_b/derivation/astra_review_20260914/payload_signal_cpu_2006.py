"""Private, CPU-only read-only audit of recorded payload signals; no policy imports."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('/home/lybm/ATEC_Experiments_20260910')
OUT = ROOT / 'task_b_astra_review_20260914/payload_signal_cpu_2006.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fit_amplitude(x, freq, dt):
    t = np.arange(len(x)) * dt
    # DC and linear trend are nuisance terms; Nyquist has no independent sine.
    cols = [np.ones(len(x)), t-t.mean(), np.cos(2*np.pi*freq*t)]
    if freq < .5/dt-1e-8:
        cols.append(np.sin(2*np.pi*freq*t))
    c = np.linalg.lstsq(np.asarray(cols).T, x, rcond=None)[0]
    return float(np.linalg.norm(c[2:]))


def stats(q, v, dt):
    duration = (len(q)-1)*dt
    fd = np.diff(q)/dt
    mean2 = (v[1:]+v[:-1])/2
    trend = np.polyval(np.polyfit(np.arange(len(q)), q, 1), np.arange(len(q)))
    w = np.hanning(len(q)); freqs = np.fft.rfftfreq(len(q), dt)
    amp = 2*np.abs(np.fft.rfft((q-trend)*w))/w.sum()
    band = (freqs >= 1.2) & (freqs <= 2.2)
    bi = np.where(band)[0]
    peak = int(bi[np.argmax(amp[band])]) if len(bi) else None
    return {
        'samples': len(q), 'endpoint_duration_s': duration,
        'q_span_rad': float(np.ptp(q)),
        'q_total_variation_rad': float(np.abs(np.diff(q)).sum()),
        'q_total_variation_per_second_rad_s': float(np.abs(np.diff(q)).sum()/duration),
        'net_displacement_per_second_rad_s': float((q[-1]-q[0])/duration),
        'q_25hz_sampled_alternating_amplitude_rad': fit_amplitude(q,.5/dt,dt),
        'v_25hz_sampled_alternating_amplitude_rad_s': fit_amplitude(v,.5/dt,dt),
        'q_fit_1p67hz_amplitude_rad': fit_amplitude(q,1.67,dt),
        'q_lowband_hann_peak_hz': None if peak is None else float(freqs[peak]),
        'q_lowband_hann_peak_amplitude_rad': None if peak is None else float(amp[peak]),
        'lowband_frequency_resolution_hz': 1/(len(q)*dt),
        'raw_velocity_abs_max_rad_s': float(abs(v).max()),
        'raw_velocity_abs_mean_rad_s': float(abs(v).mean()),
        'mean2_velocity_abs_max_rad_s': float(abs(mean2).max()),
        'finite_difference_abs_max_rad_s': float(abs(fd).max()),
        'finite_difference_abs_mean_rad_s': float(abs(fd).mean()),
        'q_step_direction_alternation_fraction': float(np.mean(fd[1:]*fd[:-1]<0)),
        'raw_velocity_sign_alternation_fraction': float(np.mean(v[1:]*v[:-1]<0)),
    }


def run(pattern):
    p = next((ROOT/'task_b_score').glob(pattern))
    n = np.load(p/'telemetry.npz'); dt = float(n['dt'])
    rows = [json.loads(line) for line in (p/'trace.jsonl').open()]
    result = json.loads((p/'result.json').read_text())
    names = n['joint_names'].tolist(); arm = [names.index(f'arm_joint{i}') for i in range(1,7)]
    q = np.asarray(n['q'][:,arm],float); v = np.asarray(n['qdot'][:,arm],float)
    mean2 = np.vstack([v[0], (v[1:]+v[:-1])/2])
    phases = np.asarray([r['policy_debug'].get('phase','') for r in rows])
    out = {'run':p.name,'source_sha256':{f:sha(p/f) for f in ['telemetry.npz','trace.jsonl','result.json','source_snapshots/task_b/payload_motion.py']},
           'dt_s':dt,'stop_reason':result['stop_reason'],'phases':{}}
    for phase in ['PAYLOAD_RAISE','PAYLOAD_SWING','PAYLOAD_EXTEND','PAYLOAD_HOLD']:
        idx=np.where(phases==phase)[0]
        if not len(idx): continue
        d = [rows[i]['policy_debug'] for i in idx]
        ix=idx[-min(201,len(idx)):]
        ent={'first_step':int(n['step'][idx[0]]),'last_step':int(n['step'][idx[-1]]),
             'endpoint_duration_s':(len(idx)-1)*dt,'raw_all_arm_abs_max_rad_s':float(abs(v[idx]).max()),
             'raw_all_arm_abs_quantiles_rad_s':np.quantile(abs(v[idx]).max(axis=1),[.5,.95,.99,1.]).tolist(),
             'mean2_all_arm_abs_max_rad_s':float(abs(mean2[idx]).max()),
             'raw_gate_failed_samples':int(np.sum(np.max(abs(v[idx]),axis=1)>.12)),
             'mean2_gate_failed_samples':int(np.sum(np.max(abs(mean2[idx]),axis=1)>.12)),
             'tail_window_steps':[int(n['step'][ix[0]]),int(n['step'][ix[-1]])],
             'tail_joint2':stats(q[ix,1],v[ix,1],dt),
             'tail_all_joint_25hz_sampled_velocity_amplitude':{name:fit_amplitude(n['qdot'][ix,i],25,dt) for i,name in enumerate(names)},
             'quiet_ready_samples':sum(bool(x.get('quiet_ready')) for x in d),
             'quiet_tick_samples':sum(bool(x.get('quiet_tick')) for x in d),
             'max_pause_s':max(x.get('pause_s',0) for x in d)}
        alpha = np.array([x.get('tracker',{}).get('path_alpha',-1) for x in d])
        settled=idx[alpha>=1-1e-12]
        if len(settled)>1: ent['after_alpha1_joint2']=stats(q[settled,1],v[settled,1],dt)
        ent['tail_26_samples_all_arm_span_rad']=float(np.ptp(q[idx[-26:]],axis=0).max()) if len(idx)>=26 else None
        if phase=='PAYLOAD_HOLD':
            sp=[]
            for end in range(idx[0]+25,idx[-1]+1):
                sp.append(float(np.ptp(q[end-25:end+1],axis=0).max()))
            ent['hold_all_complete_26sample_windows']={'count':len(sp),'passing_count':sum(s<=.002 for s in sp),
                                                        'max_span_rad':max(sp),'min_span_rad':min(sp)}
            ent['hold_quiet_tick_without_ready_samples']=sum(bool(x.get('quiet_tick')) and not bool(x.get('quiet_ready')) for x in d)
        out['phases'][phase]=ent
    payload_idx=np.where(np.char.startswith(phases,'PAYLOAD_'))[0]
    debug_v=np.asarray([rows[i]['policy_debug'].get('arm_qdot_rad_s') or [np.nan]*6 for i in payload_idx])
    out['debug_raw_qdot_vs_telemetry_max_diff']=float(np.nanmax(abs(debug_v-v[payload_idx])))
    if 'arm_qdot_denyquist_rad_s' in rows[-1]['policy_debug']:
        dd=np.asarray([rows[i]['policy_debug'].get('arm_qdot_denyquist_rad_s') or [np.nan]*6 for i in payload_idx])
        finite_rows=np.where(np.isfinite(dd).all(axis=1))[0]
        out['mean2_initialization']='First payload tick uses current twice, as no previous payload qdot exists; remaining ticks use true consecutive samples.'
        out['debug_mean2_vs_offline_max_diff_after_first_payload_tick']=float(np.nanmax(abs(dd[finite_rows[1:]]-mean2[payload_idx[finite_rows[1:]]])))
    out['stop_state_keys']=list(result['policy_stop_record']['state'])
    out['feedforward_rad']=rows[-1]['policy_debug']['tracker'].get('feedforward_rad')
    trackers=[rows[i]['policy_debug'].get('tracker') for i in payload_idx]
    trackers=[t for t in trackers if t and 'integral_trim_rad' in t]
    if trackers:
        integrals=np.asarray([t['integral_trim_rad'] for t in trackers])
        candidates=np.asarray([np.asarray(t['reference_rad'])+np.asarray(t['feedforward_rad'])+np.asarray(t['integral_trim_rad'])+np.asarray(t['filtered_correction_rad']) for t in trackers])
        commands=np.asarray([t['command_rad'] for t in trackers])
        clipped=abs(commands-candidates)>1e-7
        changing=abs(np.diff(integrals,axis=0))>1e-8
        out['integral_output_clipping_diagnostic']={
            'final_integral_rad':integrals[-1].tolist(),
            'maximum_abs_integral_rad':abs(integrals).max(axis=0).tolist(),
            'command_differs_from_candidate_samples_per_axis':clipped.sum(axis=0).tolist(),
            'integral_changes_while_command_differs_from_candidate_samples_per_axis':(clipped[1:]&changing).sum(axis=0).tolist(),
            'meaning':'Shows final output limitation is not a full anti-windup gate; does not by itself prove instability.'}
    if 'PAYLOAD_HOLD' in out['phases']:
        hi=np.where(phases=='PAYLOAD_HOLD')[0]
        sq=np.asarray(result['policy_stop_record']['state']['q'])[arm]
        hv=np.vstack([q[hi],sq])
        out['hold_including_stop_observation']={
            'phase_transition_step':int(n['step'][hi[0]]),
            'stop_observation_step':result['policy_stop_record']['before_step'],
            'transition_to_stop_elapsed_s':(result['policy_stop_record']['before_step']-int(n['step'][hi[0]]))*dt,
            'counted_eligible_samples':100,
            'counted_eligible_first_last_elapsed_s':99*dt,
            'all_arm_total_variation_per_second_rad_s':(abs(np.diff(hv,axis=0)).sum(axis=0)/((len(hv)-1)*dt)).tolist(),
            'complete_26sample_windows':len(hv)-25,
            'complete_26sample_windows_passing':int(sum(np.ptp(hv[j:j+26],axis=0).max()<=.002 for j in range(len(hv)-25)))}
    return out


runs=[run(x) for x in ['plan_p6_*','plan_p9_*','plan_p10_*','plan_p11_*','plan_p12_*','plan_p13_*']]
freq=[0,1.67,10,20,24,25]
output={'scope':'CPU read-only signal diagnostics; all arrays from saved logs, no policy writes or GPU',
        'method_notes':['Tail uses up to 201 samples (4s between endpoints). 25Hz amplitude is discrete alternating regression, not unique continuous-time spectrum.',
                        '1.2-2.2Hz Hann peak is detrended and coherent-gain normalized; short phases have coarse resolution, not modal identification.',
                        'Total variation sums all absolute consecutive changes, unlike span/net displacement.',
                        'No stop-state sample is appended here; policy stops before the final observation is put into telemetry.'],
        'mean2_transfer':{'frequency_hz':freq,'gain':[float(abs(np.cos(np.pi*f/50))) for f in freq],'group_delay_s':.01},
        'fitted_plant':{'J':.727,'K':80,'D':4,'f_undamped_hz':float(np.sqrt(80/.727)/(2*np.pi)),
                        'zeta':float(4/(2*np.sqrt(80*.727))),'f_damped_hz':float(np.sqrt(80/.727-(4/(2*.727))**2)/(2*np.pi)),
                        'inertia_is_fitted_to_observed_frequency_not_independent_prediction':True},'runs':runs}
OUT.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print(OUT)
for r in runs:
 print(r['run'])
 for phase,s in r['phases'].items():
  a=s['tail_joint2'];print(phase,s['tail_window_steps'],'rawmax',s['raw_all_arm_abs_max_rad_s'],'qspan',a['q_span_rad'],'TV/s',a['q_total_variation_per_second_rad_s'],'q25',a['q_25hz_sampled_alternating_amplitude_rad'],'v25',a['v_25hz_sampled_alternating_amplitude_rad_s'],'1.67',a['q_fit_1p67hz_amplitude_rad'],'holdwin',s.get('hold_all_complete_26sample_windows'))
 print('debug diffs',r['debug_raw_qdot_vs_telemetry_max_diff'],r.get('debug_mean2_vs_offline_max_diff_after_first_payload_tick'),'ff',r['feedforward_rad'])
