"""Collected-data follow-up; no rollouts or label changes.
Run: python -m analysis.analyze_signals
Feature choices are exploratory, informed by the original full-data screen.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from ctfeat.metrics import Bootstrap, Ranked, binary_metrics, ci
from ctfeat.models import folds
from ctfeat.preflight import check_labels

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/study_qwen25_7b_math500_exhaustive/dataset'
FEATURES = ROOT / 'outputs/signals/features'
OUT = ROOT / 'analysis/results/signals_audit'


def load_frame():
    check_labels(DATA)
    p = pd.read_parquet(DATA / 'positions.parquet')
    t = pd.read_json(DATA / 'traces.jsonl', lines=True)
    f = p.merge(t[['trace_id', 'problem_id', 'correct', 'crit_drop', 'crit_cdpo']], on='trace_id', validate='many_to_one')
    t1 = pd.read_parquet(FEATURES / 'tier1.parquet')
    f = f.merge(t1, on=['trace_id', 'pos'], validate='one_to_one')
    extras = ['dla_total__pre', 'dla_conflict_frac__pre', 'dla_conflict__pre', 'phi_tok_raw__pre',
              'phi_tok_ref__pre', 'phi_tok_prompt__pre', 'phi_win_64__pre', 'phi_win_32__pre']
    f = f.merge(pd.read_parquet(FEATURES / 'tier3_derived.parquet', columns=['trace_id', 'pos'] + extras),
                on=['trace_id', 'pos'], validate='one_to_one')
    alt_p = np.where(f['rank_chosen__pre'] == 1, f['pmax__pre'] - f['margin12__pre'], f['pmax__pre'])
    f['negative_logit_gap'] = np.log(np.maximum(alt_p, 1e-30)) - f['logp_chosen__pre']
    f['surprisal'] = -f['logp_chosen__pre']
    f['non_argmax'] = (f['rank_chosen__pre'] > 1).astype(float)
    f['A'] = f.is_crit_drop.astype(bool)
    f['C'] = f.is_crit_cdpo.astype(bool)
    # Sensitivity target only; does not replace or modify saved labels.
    f['S'] = (~f.correct) & (f.pos == f.crit_drop)
    f = f.sort_values(['trace_id', 'pos']).reset_index(drop=True)
    f['att_resp_L25_H3__post'] = np.nan
    meta = json.loads((FEATURES / 'extract_meta.json').read_text())
    stat = meta['attn_stats'].index('att_resp')
    for tid, group in f.groupby('trace_id', sort=False):
        safe = tid.replace(':', '_').replace('/', '_')
        arr = np.load(FEATURES / 'attn' / f'{safe}.npy', mmap_mode='r')
        f.loc[group.index, 'att_resp_L25_H3__post'] = arr[group.pos.to_numpy() + 1, 24, 3, stat].astype(float)
    return f, [c for c in t1 if c.endswith('__pre') and c != 'warmup__pre']


def evaluate_score(y, score, groups, boot):
    good = np.isfinite(score)
    y, score, groups = y[good].astype(float), score[good], groups[good]
    ranked = Ranked.build(score)
    point = binary_metrics(ranked, y, np.ones(len(y)))
    assert abs(point['auroc'] - roc_auc_score(y, score)) < 1e-10
    assert abs(point['auprc'] - average_precision_score(y, score)) < 1e-10
    samples = np.array([list(binary_metrics(ranked, y, w).values()) for w in boot.weights(groups)])
    row = dict(n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()))
    for i, key in enumerate(point):
        row[key] = point[key]
        row[key + '_lo'], row[key + '_hi'] = ci(samples[:, i])
    return row, samples


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f, t1 = load_frame()
    wrong = (~f.correct).to_numpy()
    groups = f.problem_id.astype(str).to_numpy()
    splits = folds(groups, 5, 0)
    boot = Bootstrap.draw(groups, 1000, 42)
    direct = {'entropy': f['ent__pre'].to_numpy(), 'negative_pmax': -f['pmax__pre'].to_numpy(),
              'surprisal': f.surprisal.to_numpy(), 'negative_logit_gap': f.negative_logit_gap.to_numpy(),
              'negative_dla_total': -f['dla_total__pre'].to_numpy(),
              'dla_conflict_frac': f['dla_conflict_frac__pre'].to_numpy(), 'non_argmax': f.non_argmax.to_numpy(),
              'negative_phi_raw': -f['phi_tok_raw__pre'].to_numpy(),
              'negative_attention_response_post': -f['att_resp_L25_H3__post'].to_numpy()}
    phi = ['phi_tok_raw__pre', 'phi_tok_ref__pre', 'phi_tok_prompt__pre', 'phi_win_64__pre', 'phi_win_32__pre']
    sets = {'T1_fixed': t1, 'T1_gap': t1 + ['negative_logit_gap'],
            'T1_gap_conflict': t1 + ['negative_logit_gap', 'dla_conflict_frac__pre'],
            'T1_gap_phi': t1 + ['negative_logit_gap'] + phi,
            'T1_gap_attention_post': t1 + ['negative_logit_gap', 'att_resp_L25_H3__post']}
    rows, gains, thresholds_out, subgroups, hits_out = [], [], [], [], []
    predictions = f[['trace_id', 'pos', 'problem_id', 'correct', 'A', 'C', 'S']].copy()
    all_scores = {}
    for target in ('A', 'C', 'S'):
        y = f[target].to_numpy(bool)
        scores = dict(direct)
        for name, cols in sets.items():
            X = f[cols].to_numpy(dtype=float, copy=True)
            X[np.isposinf(X)] = 29
            X[np.isneginf(X)] = np.nan
            score = np.full(len(f), np.nan)
            for tr, te in splits:
                tr = tr[wrong[tr]]
                assert not set(groups[tr]) & set(groups[te])
                pipe = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
                                     LogisticRegression(C=.1, class_weight='balanced', max_iter=3000))
                pipe.fit(X[tr], y[tr])
                score[te] = pipe.decision_function(X[te])
            assert np.isfinite(score).all()
            scores[name] = score
        samples_by_name = {}
        for name, score in scores.items():
            row, samples = evaluate_score(y[wrong], score[wrong], groups[wrong], boot)
            rows.append(dict(target=target, score=name, **row))
            samples_by_name[name] = samples
            predictions[f'{target}:{name}'] = score
            print(target, name, {k: round(row[k], 5) for k in ('auroc', 'auprc', 'fa_per_1000_at_50recall')}, flush=True)
            pools = {'no_engine_mismatch': wrong & ~f.engine_mismatch.fillna(False).to_numpy(bool),
                     'non_argmax_only': wrong & (f.non_argmax.to_numpy() > 0),
                     'entropy_top_quintile': wrong & (f['ent__pre'].to_numpy() >= f['ent__pre'].quantile(.8)),
                     'all_wrong_plus_right': np.ones(len(f), bool),
                     'noncritical': wrong & (y | f.is_noncritical.to_numpy(bool))}
            for pool, mask in pools.items():
                m = mask & np.isfinite(score)
                met = binary_metrics(Ranked.build(score[m]), y[m].astype(float), np.ones(m.sum()))
                subgroups.append(dict(target=target, score=name, pool=pool, n_pos=int(y[m].sum()), n_neg=int((~y[m]).sum()), **met))
            hits = []
            for tid, g in f[wrong].groupby('trace_id'):
                ix = g.index.to_numpy()
                if y[ix].any():
                    order = ix[np.argsort(-score[ix], kind='stable')]
                    hits.append((bool(y[order[:1]].any()), bool(y[order[:5]].any())))
            hits_out.append(dict(target=target, score=name, traces=len(hits), top1=np.mean([h[0] for h in hits]), top5=np.mean([h[1] for h in hits])))
            # Train-only threshold calibration for fixed-form scores. Fitted scores
            # would need inner OOF calibration and are intentionally omitted here.
            if name in ('surprisal', 'negative_logit_gap', 'dla_conflict_frac', 'entropy'):
                calls = np.zeros(len(f), bool)
                thresholds = []
                for tr, te in splits:
                    tr = tr[wrong[tr]]
                    threshold = np.quantile(score[tr][y[tr]], .5, method='lower')
                    calls[te] = score[te] >= threshold
                    thresholds.append(float(threshold))
                tp = int((calls & y & wrong).sum()); fp = int((calls & ~y & wrong).sum())
                thresholds_out.append(dict(target=target, score=name, recall=tp/y[wrong].sum(), precision=tp/(tp+fp),
                                           fa_per_1000=1000*fp/(wrong & ~y).sum(),
                                           right_trace_alarms_per_1000=1000*calls[~wrong].mean(), thresholds=json.dumps(thresholds)))
        comparisons = [('negative_logit_gap', 'surprisal'), ('dla_conflict_frac', 'negative_logit_gap'),
                       ('T1_gap', 'T1_fixed'), ('T1_gap_conflict', 'T1_gap'), ('T1_gap_phi', 'T1_gap'),
                       ('T1_gap_attention_post', 'T1_gap')]
        for name, baseline in comparisons:
            for j, metric in enumerate(('auroc', 'auprc', 'fa_per_1000_at_50recall')):
                low, high = ci(samples_by_name[name][:, j] - samples_by_name[baseline][:, j])
                cur = next(r for r in rows if r['target']==target and r['score']==name)
                base = next(r for r in rows if r['target']==target and r['score']==baseline)
                gains.append(dict(target=target, score=name, baseline=baseline, metric=metric,
                                  difference=cur[metric]-base[metric], ci_low=low, ci_high=high))
        all_scores[target] = scores
    for filename, records in [('scores', rows), ('gains', gains), ('thresholds', thresholds_out), ('subgroups', subgroups), ('trace_localization', hits_out)]:
        pd.DataFrame(records).to_csv(OUT / f'{filename}.csv', index=False)
    predictions.to_parquet(OUT / 'predictions.parquet', index=False)
    gap = f.negative_logit_gap.to_numpy()[wrong]
    dla = -f['dla_total__pre'].to_numpy()[wrong]
    S = f['dla_total__pre'].to_numpy(); N = f['dla_conflict__pre'].to_numpy(); frac = f['dla_conflict_frac__pre'].to_numpy()
    audit = {'n_rows': len(f), 'n_wrong': int(wrong.sum()), 'n_problems': f.problem_id.nunique(),
             'targets': {t: {'positives': int(f.loc[wrong,t].sum()),
                             'problems': f.loc[wrong & f[t].to_numpy(bool),'problem_id'].nunique(),
                             'traces': f.loc[wrong & f[t].to_numpy(bool),'trace_id'].nunique(),
                             'low_entropy_positives': int((wrong & f[t].to_numpy(bool) & (f['ent__pre']<f['ent__pre'].median()).to_numpy()).sum())} for t in ('A','C','S')},
             'A_S_overlap': int((f.A & f.S).sum()), 'A_C_overlap': int((f.A & f.C).sum()), 'S_C_overlap': int((f.S & f.C).sum()),
             'gap_dla_spearman': float(spearmanr(gap,dla).statistic),
             'gap_dla_abs_error_quantiles': np.quantile(abs(gap-dla),[.5,.9,.99,1]).tolist(),
             'conflict_identity_max_error': float(np.nanmax(abs(frac - .5*(1-S/(S+2*N))))),
             'non_argmax_shares': {t: {'positive': float(f.loc[wrong & f[t].to_numpy(bool),'non_argmax'].mean()),
                                     'negative': float(f.loc[wrong & ~f[t].to_numpy(bool),'non_argmax'].mean())} for t in ('A','C','S')},
             'cv': '5 problem-grouped folds, seed 0; fixed C=0.1; balanced classes; train-only imputation/scaling',
             'bootstrap': '1000 paired problem resamples, seed 42; fixed OOF predictions, no model refitting', 'features': sets}
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for ax, target, title in zip(axes, ('A','S','C'), ('Large-drop labels (109)', 'Sharpest event per trace (200)', 'cDPO (200)')):
        y=f[target].to_numpy(bool)[wrong]
        for name in ('entropy','surprisal','negative_logit_gap','dla_conflict_frac','T1_gap_attention_post'):
            precision, recall, _=precision_recall_curve(y, all_scores[target][name][wrong])
            ax.plot(recall,precision,label=name)
        ax.axhline(y.mean(), color='gray',linestyle=':')
        ax.set(xlabel='Recall',ylabel='Precision',title=title,xlim=(0,1),ylim=(0,.35))
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7)
    fig.suptitle('Collected-data audit; exploratory additions; precision axis capped at 0.35')
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'precision_recall.{ext}',dpi=180)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
