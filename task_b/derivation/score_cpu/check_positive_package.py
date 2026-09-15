"""Negative packaging tests only; no synthetic positive run is constructed."""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path('/home/lybm/ATEC_Robotics_Projects_20260910')
REAL_ZERO = Path('/home/lybm/ATEC_Experiments_20260910/task_b_score/p0_evaluator_seed42_01')
TOOL = REPO / 'task_b/package_positive.py'
AUDITOR = REPO / 'task_b/audit_positive.py'
spec = importlib.util.spec_from_file_location('pack', TOOL)
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)
checks = []


def rejected(name, run, out, auditor=AUDITOR, contains=None):
    before_exists = out.exists()
    proc = subprocess.run([sys.executable, str(TOOL), str(run), str(out),
                           '--auditor', str(auditor)], text=True, capture_output=True)
    passed = proc.returncode == 2 and out.exists() == before_exists
    if contains:
        passed = passed and contains in proc.stderr
    checks.append({'name': name, 'passed': passed, 'exit': proc.returncode,
                   'message': proc.stderr.strip(), 'output_directory_created': not before_exists and out.exists()})


with tempfile.TemporaryDirectory(prefix='taskb-package-negative-') as work:
    temp = Path(work)
    rejected('real_zero_fresh_audit', REAL_ZERO, temp / 'zero_package',
             contains='not_achieved_zero_score')
    rejected('real_first_reach_shared_geometry_snapshot_accepted_then_zero_rejected',
             REAL_ZERO.parent / 'first_reach_side_seed42_01', temp / 'side_package',
             contains='not_achieved_zero_score')
    run = temp / 'copied_zero'
    shutil.copytree(REAL_ZERO, run)
    (run / 'independent_positive_audit.json').write_text(json.dumps({'verified_positive_raw_score': True}))
    rejected('cached_positive_boolean_is_ignored', run, temp / 'cache_package',
             contains='not_achieved_zero_score')
    marker = run / 'SYNTHETIC_ONLY.txt'
    marker.write_text('Synthetic boundary test; underlying score remains zero.\n')
    rejected('explicit_synthetic_marker_rejected', run, temp / 'synthetic_package',
             contains='Synthetic audit fixtures')
    marker.unlink()
    result_path = run / 'result.json'
    real_result = result_path.read_bytes()
    marked_result = json.loads(real_result)
    marked_result['fixture_notice'] = 'Synthetic boundary test; score remains zero.'
    result_path.write_text(json.dumps(marked_result))
    rejected('explicit_fixture_notice_rejected', run, temp / 'notice_package',
             contains='Synthetic fixture_notice')
    result_path.write_bytes(real_result)
    fake = temp / 'fake_auditor.py'
    fake.write_text('print("{\\"verified_positive_raw_score\\": true}")\n')
    rejected('unreviewed_auditor_cannot_substitute_boolean', REAL_ZERO, temp / 'fake_package', fake,
             contains='SHA-256 mismatch')
    existing = temp / 'existing'
    existing.mkdir()
    (existing / 'sentinel').write_text('untouched')
    rejected('existing_output_is_not_overwritten', REAL_ZERO, existing,
             contains='Output must be a new directory')
    checks.append({'name': 'existing_sentinel_unchanged',
                   'passed': (existing / 'sentinel').read_text() == 'untouched'})
    rejected('output_inside_input_rejected', run, run / 'new_package',
             contains='Output must be outside')
    manifest_file = run / 'source_manifest.json'
    original_manifest = manifest_file.read_bytes()
    manifest = json.loads(original_manifest)
    entries = manifest['hashed_files']
    key = next(k for k, v in entries.items() if v['root'] == 'project')
    snapshot = run / 'source_snapshots' / entries[key]['relative_path']
    snapshot_bytes = snapshot.read_bytes()
    snapshot.write_bytes(snapshot_bytes + b'\n# corrupted fixture\n')
    rejected('snapshot_hash_tampering_rejected', run, temp / 'tamper_package',
             contains='Snapshot does not match')
    snapshot.write_bytes(snapshot_bytes)
    outside = temp / 'outside.py'
    outside.write_bytes(snapshot_bytes)
    snapshot.unlink()
    snapshot.symlink_to(outside)
    rejected('snapshot_symlink_escape_rejected', run, temp / 'link_package',
             contains='Symlink artifact')
    snapshot.unlink()
    snapshot.write_bytes(snapshot_bytes)
    entries[key]['relative_path'] = 'task_b/../../../outside.py'
    manifest_file.write_text(json.dumps(manifest))
    rejected('manifest_parent_traversal_rejected', run, temp / 'traversal_package',
             contains='Unsafe artifact path')
    manifest_file.write_bytes(original_manifest)
    trace = run / 'trace.jsonl'
    trace_bytes = trace.read_bytes()
    outside_trace = temp / 'outside_trace.jsonl'
    outside_trace.write_bytes(trace_bytes)
    trace.unlink()
    trace.symlink_to(outside_trace)
    rejected('raw_trace_symlink_escape_rejected', run, temp / 'trace_link_package',
             contains='Symlink artifact')
    trace.unlink()
    trace.write_bytes(trace_bytes)
    # Empty or Boolean-only documents must fail the report validation itself.
    try:
        pack.verify_report({'verified_positive_raw_score': True},
                           json.loads((run / 'result.json').read_text()), run.resolve())
        passed = False
    except pack.Rejected:
        passed = True
    checks.append({'name': 'boolean_only_report_rejected', 'passed': passed})

report = {'scope': 'Real zero-score rejection plus negative filesystem/report fixtures; no positive score fabricated.',
          'tool_sha256': pack.digest(TOOL), 'auditor_sha256': pack.digest(AUDITOR),
          'real_zero_run': str(REAL_ZERO), 'checks': checks,
          'passed': all(item['passed'] for item in checks),
          'positive_packaging_verified': False,
          'limitation': 'The successful packaging branch still requires a real independently verified positive run.'}
output = Path(__file__).with_suffix('.json')
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'report': str(output), 'checks': len(checks), 'passed': report['passed']}))
raise SystemExit(0 if report['passed'] else 1)
