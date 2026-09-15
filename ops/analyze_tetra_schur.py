#!/usr/bin/env python3
"""Audit retained pilot traces against dense H and reserved activations, then summarize.

Usage: uv run --offline --with numpy python ops/analyze_tetra_schur.py PILOT_DIR OUTPUT.json
No checkpoint loading, model inference, gain selection or new experiment is performed.
"""
import argparse
import json
import pathlib
import numpy as np


def read(path):
    return json.loads(path.read_text())


def close(actual, expected, name):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValueError(f"non-finite {name}")
    error = float(np.max(np.abs(actual-expected)))
    scale = max(float(np.max(np.abs(expected))), 1e-15)
    if error > 2e-9*scale + 2e-13:
        raise ValueError(f"{name}: max error {error}, scale {scale}")
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    args = parser.parse_args()
    events = [json.loads(line) for line in (args.pilot/'run-events.jsonl').read_text().splitlines()]
    assert events[-1]['kind'] == 'complete'
    assert sum(e['kind'] == 'step_end' and e['exit_code'] == 0 for e in events) == 14
    cells, changed, seed_reports = [], [], []
    audited_branches = audited_snapshots = 0
    maxima = dict(loss=0.0, validation=0.0, conditional=0.0, prefix=0.0, gradient=0.0)
    for seed in [1, 2]:
        seed_dir = args.pilot/f'seed-{seed}'
        assert (seed_dir/'capture/capture-complete.json').exists()
        oracle = read(seed_dir/'capture/oracle.json')
        assert oracle['relative'] < oracle['threshold']
        seed_cells = []
        for layer in [0, 13, 27]:
            for projection in ['self_attn.q_proj', 'mlp.gate_proj']:
                stem = f"layer-{layer}-{projection.replace('.', '-')}"
                result_dir = seed_dir/stem
                summary = read(result_dir/'summary.json')
                bundle = read(seed_dir/'capture'/f'{stem}.json')
                assert bundle['projection'] == projection and bundle['layer'] == layer
                n = bundle['width']
                assert n == 1024
                def array(key):
                    meta = bundle[key]
                    a = np.fromfile(seed_dir/'capture'/meta['name'], dtype='<f8')
                    assert len(a) == meta['count'] and np.isfinite(a).all()
                    return a
                h = array('hessian').reshape(n, n)
                h = h + bundle['plan']['damping']*np.trace(h)/n*np.eye(n)
                validation = array('validation').reshape(-1, n)
                original = array('original_rows').reshape(-1, n)
                # Direct Schur complements, independent of the Rust inverse-Hessian factor.
                schur = {}
                for block in bundle['snapshot_blocks']:
                    start, end = block*24, (block+1)*24
                    s = h[start:end, start:end].copy()
                    if end < n:
                        s -= h[start:end, end:] @ np.linalg.solve(h[end:, end:], h[end:, start:end])
                    schur[block] = s
                regrets = np.zeros(3)
                val_regrets = np.zeros(3)
                disagreements = np.zeros(3, dtype=int)
                shadow_count = snapshots = 0
                positional = {}
                for row_index, row_id in enumerate(bundle['row_ids']):
                    doc = read(result_dir/f'row-{row_id}.json')['diagnostic']
                    w = original[row_index]
                    close(doc['original'], w, 'original row')
                    assert len(doc['shadow']) == n//24
                    shadow_count += len(doc['shadow'])
                    for shadow in doc['shadow']:
                        a, b, c = shadow['choices_ABC']
                        disagreements += [a != b, a != c, b != c]
                    assert [s['comparison']['block'] for s in doc['snapshots']] == bundle['snapshot_blocks']
                    for snapshot in doc['snapshots']:
                        snapshots += 1
                        audited_snapshots += 1
                        c = snapshot['comparison']
                        start, end = c['block']*24, (c['block']+1)*24
                        working = np.asarray(snapshot['working'])
                        ebase = working-w
                        gradient = h @ ebase
                        maxima['gradient'] = max(maxima['gradient'], close(gradient[start:], np.zeros(n-start), 'conditional center gradient'))
                        prefix = float(ebase @ h @ ebase)
                        maxima['prefix'] = max(maxima['prefix'], close(c['prefix_loss'], prefix, 'fixed-prefix constant'))
                        q = np.asarray([b['reconstructed'] for b in snapshot['branches']])
                        e = q-w
                        losses = np.einsum('ij,ij->i', e @ h, e)
                        val = np.mean((validation @ e.T)**2, axis=0)
                        maxima['loss'] = max(maxima['loss'], close([b['rollout_loss'] for b in snapshot['branches']], losses, 'full quadratic loss'))
                        maxima['validation'] = max(maxima['validation'], close([b['validation_loss'] for b in snapshot['branches']], val, 'reserved projection output loss'))
                        for g, branch in enumerate(snapshot['branches']):
                            audited_branches += 1
                            assert branch['gain'] == g
                            residual = working[start:end]-q[g, start:end]
                            local = float(residual @ schur[c['block']] @ residual)
                            maxima['conditional'] = max(maxima['conditional'], close(c['conditional'][g], local, 'direct Schur score'))
                            close(branch['continuous_lower_bound'], prefix+local, 'global bound')
                            close(branch['rollout_excess'], losses[g]-prefix-local, 'rollout excess')
                            assert branch['rollout_excess'] >= -2e-9*max(abs(losses[g]), 1.0)
                            for b, code in enumerate(branch['codes']):
                                point = np.asarray(code['point'], dtype=float)
                                norm = np.linalg.norm(point)
                                decoded = point*(bundle['centroids'][code['gain']]*doc['row_scale']/norm) if norm else point
                                close(q[g, b*24:(b+1)*24], decoded, 'code reconstruction')
                        choices = c['choices_ABC']
                        current = np.asarray([snapshot['branches'][g]['rollout_loss'] for g in choices])
                        current_val = np.asarray([snapshot['branches'][g]['validation_loss'] for g in choices])
                        best = min(b['rollout_loss'] for b in snapshot['branches'])
                        best_val = min(b['validation_loss'] for b in snapshot['branches'])
                        regrets += current-best
                        val_regrets += current_val-best_val
                        part = positional.setdefault(c['block'], dict(count=0, changes_BC=0, rollout_delta_CB=0., validation_delta_CB=0.))
                        part['count'] += 1
                        part['changes_BC'] += choices[1] != choices[2]
                        part['rollout_delta_CB'] += float(current[2]-current[1])
                        part['validation_delta_CB'] += float(current_val[2]-current_val[1])
                        if choices[1] != choices[2]:
                            changed.append(dict(seed=seed, layer=layer, projection=projection, row=row_id, block=c['block'],
                                choices_ABC=choices, rollout_delta_CB=float(current[2]-current[1]),
                                validation_delta_CB=float(current_val[2]-current_val[1]),
                                future_full_blocks=n//24-c['block']-1))
                close(summary['mean_rollout_regret_ABC'], regrets/snapshots, 'summary rollout regret')
                close(summary['mean_validation_regret_ABC'], val_regrets/snapshots, 'summary validation regret')
                assert disagreements.tolist() == summary['disagreements_AB_AC_BC']
                assert summary['snapshot_count'] == snapshots == 12
                assert summary['shadow_count'] == shadow_count == 168
                cell = dict(seed=seed, layer=layer, projection=projection, rows=4, shadow_count=shadow_count,
                    disagreements_AB_AC_BC=disagreements.tolist(), snapshot_count=snapshots,
                    mean_rollout_regret_ABC=(regrets/snapshots).tolist(),
                    mean_validation_regret_ABC=(val_regrets/snapshots).tolist(),
                    positional_audit_exploratory=positional)
                cells.append(cell)
                seed_cells.append(cell)
        agg = dict(seed=seed, cell_count=len(seed_cells), shadow_count=sum(c['shadow_count'] for c in seed_cells),
            disagreements_AB_AC_BC=np.sum([c['disagreements_AB_AC_BC'] for c in seed_cells],axis=0).tolist(),
            changed_snapshot_count=sum(c['seed']==seed for c in changed))
        for kind in ['rollout','validation']:
            means = np.mean([c[f'mean_{kind}_regret_ABC'] for c in seed_cells],axis=0)
            agg[f'mean_{kind}_regret_ABC'] = means.tolist()
            agg[f'{kind}_relative_reduction_C_vs_B_percent'] = float(100*(1-means[2]/means[1])) if means[1] else None
            agg[f'{kind}_relative_reduction_B_vs_A_percent'] = float(100*(1-means[1]/means[0])) if means[0] else None
        seed_reports.append(agg)
    report = dict(label='computed from measured retained outputs; no new model run',
        seconds_measured=events[-1]['elapsed_seconds'],remote_cost_usd=0,
        audited_snapshots=audited_snapshots,audited_branches=audited_branches,
        independent_dense_audit_max_absolute_errors=maxima,
        cells=cells,seeds=seed_reports,changed_BC_snapshots=changed,
        positional_audit_note='Descriptive post-run audit; no new exclusion or adoption criterion. Last full blocks have only a continuous KeepExact tail.')
    with args.output.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({k:report[k] for k in ['seconds_measured','audited_snapshots','audited_branches','independent_dense_audit_max_absolute_errors','seeds','changed_BC_snapshots']},indent=2))


if __name__ == '__main__':
    main()
