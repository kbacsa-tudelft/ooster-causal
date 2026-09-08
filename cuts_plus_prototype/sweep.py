"""
Hyperparameter sweep runner for cuts_plus_rca.py.

Runs the pipeline once per combination in the sweep grid. Each combination gets its own
--log-dir/--save-dir subdirectory under one shared root, so all runs land as separate TensorBoard
"runs" under a single logdir and can be compared side by side:

    tensorboard --logdir cuts_plus_prototype/runs/sweep/<sweep_id>

Each combination runs in its own subprocess (same isolation convention used elsewhere in this repo
for benchmarking) - so a crash in one combination doesn't take down the rest, and peak-memory
reporting per run stays meaningful.

Usage:
    # capacity sweep, real data
    python3 cuts_plus_prototype/sweep.py --data-dir /path/to/data --total-epoch 30 \\
        --sweep mlp_hid=32,64,128 --sweep gru_layers=1,2

    # any CUTSPlusRCAConfig field not listed in --sweep is held fixed at its CLI/default value
    python3 cuts_plus_prototype/sweep.py --sweep lr_data_start=1e-2,1e-3 --dry-run

    # resume a sweep that stopped partway through (same --sweep grid must be passed again so the
    # combination list matches) - skips any combination whose final artifact already exists
    python3 cuts_plus_prototype/sweep.py --data-dir /path/to/data --total-epoch 30 \\
        --sweep mlp_hid=32,64,128 --sweep gru_layers=1,2 --resume

    # or target a specific sweep id explicitly instead of "most recent"
    python3 cuts_plus_prototype/sweep.py ... --resume 20260907_223125
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from dataclasses import fields

# Under `nohup ... > sweep.log &`, stdout is redirected to a file, so Python switches from
# line-buffered to fully-buffered (~8KB) - a short sweep like "Sweeping N combinations..." can sit
# unflushed for a long time even though everything is actually running fine. Force line buffering so
# progress is visible in the log immediately.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cuts_plus_rca import CUTSPlusRCAConfig, build_arg_parser  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUTS_PLUS_RCA = os.path.join(SCRIPT_DIR, 'cuts_plus_rca.py')


def parse_sweep_grid(sweep_args):
    """--sweep name=v1,v2,v3 (repeatable) -> {name: [v1, v2, v3]}, values cast to the field's type."""
    field_types = {f.name: f.type for f in fields(CUTSPlusRCAConfig)}
    grid = {}
    for spec in sweep_args:
        if '=' not in spec:
            raise ValueError(f'--sweep expects name=v1,v2,... - got {spec!r}')
        name, values_str = spec.split('=', 1)
        if name not in field_types:
            raise ValueError(f'Unknown CUTSPlusRCAConfig field: {name!r}')
        cast = field_types[name]
        grid[name] = [cast(v) for v in values_str.split(',')]
    return grid


def run_name(combo):
    return '_'.join(f'{k}={v}' for k, v in combo.items())


def is_done(run_dir):
    """A combination is considered complete once its final artifact exists - both run_pipeline and
    run_real_data_pipeline in cuts_plus_rca.py save this as one of their last actions, so its
    presence means that combo's process ran to completion rather than crashing/getting killed
    mid-training."""
    return os.path.exists(os.path.join(run_dir, 'models', 'cuts_plus_graph.npy'))


def build_command(base_config: CUTSPlusRCAConfig, combo: dict, run_dir: str):
    overrides = {f.name: getattr(base_config, f.name) for f in fields(base_config)}
    overrides.update(combo)
    overrides['log_dir'] = run_dir
    overrides['save_dir'] = os.path.join(run_dir, 'models')

    cli_args = []
    for k, v in overrides.items():
        cli_args += [f'--{k.replace("_", "-")}', str(v)]
    return [sys.executable, CUTS_PLUS_RCA, *cli_args]


def main():
    parser = build_arg_parser()
    parser.add_argument('--sweep', action='append', default=[],
                         help='name=v1,v2,... (repeatable) - CUTSPlusRCAConfig field(s) to sweep over')
    parser.add_argument('--sweep-root', default=os.path.join('runs', 'sweep'),
                         help='parent directory for this sweep\'s per-run subdirectories')
    parser.add_argument('--dry-run', action='store_true', help='print the planned runs and exit')
    parser.add_argument('--resume', nargs='?', const='__latest__', default=None, metavar='SWEEP_ID',
                         help='continue an existing sweep instead of starting a new one: skips any '
                              'combination that already completed. Pass a sweep id (the timestamped '
                              'directory name under --sweep-root) to target it explicitly, or bare '
                              '--resume to pick up the most recent sweep under --sweep-root.')
    args = parser.parse_args()

    grid = parse_sweep_grid(args.sweep)
    if not grid:
        print('No --sweep given; nothing to do. Example: --sweep mlp_hid=32,64,128')
        sys.exit(1)

    base_overrides = {k: v for k, v in vars(args).items()
                       if k not in ('sweep', 'sweep_root', 'dry_run', 'resume')}
    base_config = CUTSPlusRCAConfig(**base_overrides)

    names = list(grid.keys())
    combos = [dict(zip(names, values)) for values in itertools.product(*grid.values())]

    if args.resume:
        if args.resume == '__latest__':
            existing = sorted(d for d in os.listdir(args.sweep_root)
                               if os.path.isdir(os.path.join(args.sweep_root, d))) \
                if os.path.isdir(args.sweep_root) else []
            if not existing:
                print(f'--resume: no existing sweeps found under {args.sweep_root}')
                sys.exit(1)
            sweep_id = existing[-1]
        else:
            sweep_id = args.resume
        sweep_dir = os.path.join(args.sweep_root, sweep_id)
        if not os.path.isdir(sweep_dir):
            print(f'--resume: no such sweep directory: {sweep_dir}')
            sys.exit(1)
        print(f'Resuming sweep: {sweep_dir}')
    else:
        sweep_id = time.strftime('%Y%m%d_%H%M%S')
        sweep_dir = os.path.join(args.sweep_root, sweep_id)

    print(f'Sweeping {len(combos)} combination(s) over {names}')
    print(f'TensorBoard root: {sweep_dir}')
    for combo in combos:
        print(f'  {run_name(combo)}')

    if args.dry_run:
        return

    results = []
    for i, combo in enumerate(combos, 1):
        name = run_name(combo)
        run_dir = os.path.join(sweep_dir, name)
        log_path = os.path.join(run_dir, 'stdout.log')

        if args.resume and is_done(run_dir):
            print(f'\n=== [{i}/{len(combos)}] {name} === already completed, skipping')
            results.append({'name': name, 'ok': True, 'time_s': 0.0, 'log': log_path, 'skipped': True})
            continue

        os.makedirs(run_dir, exist_ok=True)
        cmd = build_command(base_config, combo, run_dir)

        print(f'\n=== [{i}/{len(combos)}] {name} ===')
        start = time.time()
        with open(log_path, 'w') as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        elapsed = time.time() - start

        ok = proc.returncode == 0
        status = 'ok' if ok else f'FAILED (exit {proc.returncode}, see {log_path})'
        print(f'    {status} - {elapsed:.1f}s')
        results.append({'name': name, 'ok': ok, 'time_s': elapsed, 'log': log_path, 'skipped': False})

    print('\n' + '=' * 78)
    print(f'{"run":50s} {"status":18s} {"time(s)":>9s}')
    for r in results:
        status = 'skipped (resumed)' if r.get('skipped') else ('yes' if r['ok'] else 'NO')
        print(f'{r["name"]:50s} {status:18s} {r["time_s"]:9.1f}')
    print(f'\nCompare all runs: tensorboard --logdir {sweep_dir}')


if __name__ == '__main__':
    main()
