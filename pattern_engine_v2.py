"""
pattern_engine_v2.py
--------------------
Automated pattern detection engine that mimics manual Futures First research analysis.
Each detected pattern is a structured 'finding' dict rendered as a tagged card.

Pattern types detected:
 1  Period Directional Bias       - overall UP/DOWN direction across years
 2  Consecutive N-Day Run         - 2/3/4 consecutive same-direction days
 3  Day-Follows-Day               - BD(n+1) repeats BD(n) direction  
 4  First-Day Predictor           - BD1 direction == period direction
 5  Pre-Roll Volume Spike         - volume elevated on last P1 day (pre-Goldman)
 6  Volume Concentration          - roll volume front-loaded vs back-loaded
 7  Peak Volume BD                - which BD has peak volume, consistently
 8  Rogers-Goldman Reversal       - P2 direction flips P1 direction
 9  P2-P3 Continuation            - P3 continues P2 direction
10  OI Direction Pattern          - OI consistently rising/falling in period
11  OI-Price Classification       - new longs / new shorts / covering / liquidation
12  Momentum Statistics           - avg range, move, IQR for the period
"""

import pandas as pd
import numpy as np
from collections import Counter

SERIES_LABELS = {
    'out': 'Outright (CC H)',
    'hk':  'H-K Spread',
    'kn':  'K-N Spread',
    'fly': 'HKN Fly',
}
PERIOD_LABELS = {
    'P1':    'Period 1 (Post-LTD)',
    'P2':    'Goldman Roll',
    'P3':    'Post-Roll → Expiry',
    'P1→P2': 'P1 → P2 transition',
    'P2→P3': 'P2 → P3 transition',
}
PTYPE_COLORS = {
    'Period Directional Bias':   '#388bfd',
    'Consecutive 2-Day Run':     '#3fb950',
    'Consecutive 3-Day Run':     '#2ea043',
    'Consecutive 4-Day Run':     '#1a7431',
    'Day-Follows-Day':           '#56d364',
    'First-Day Predictor':       '#bc8cff',
    'Pre-Roll Volume Spike':     '#f0883e',
    'Volume Concentration':      '#ffa657',
    'Peak Volume BD':            '#ffb347',
    'Rogers-Goldman Reversal':   '#da3633',
    'P2-P3 Continuation':        '#ff7b72',
    'OI Direction Pattern':      '#388bfd',
    'OI-Price Classification':   '#79c0ff',
    'Momentum Statistics':       '#8b949e',
}

def _conf(hr):
    if hr >= 0.75:  return 'HIGH'
    if hr >= 0.625: return 'MEDIUM'
    return 'LOW'

def _f(series, period, ptype, tag, desc, hits, n, yhit, ymiss,
       avg_move=None, move_range=None, bd_start=None, bd_end=None,
       vol_ctx=None, oi_ctx=None, note=None):
    hr = hits / n if n > 0 else 0
    return dict(
        series=series,
        series_label=SERIES_LABELS.get(series, series),
        period=period,
        period_label=PERIOD_LABELS.get(period, period),
        pattern_type=ptype,
        ptype_color=PTYPE_COLORS.get(ptype, '#8b949e'),
        tag=tag,
        description=desc,
        hit_rate=hr,
        hits=hits,
        n_years=n,
        years_hit=sorted(yhit),
        years_miss=sorted(ymiss),
        confidence=_conf(hr),
        avg_move=avg_move,
        move_range=move_range,
        bd_start=bd_start,
        bd_end=bd_end,
        volume_context=vol_ctx,
        oi_context=oi_ctx,
        additional_note=note,
    )

def _period_data(data, years, series, col, period):
    out = {}
    for yr in [y for y in years if y in data]:
        sub = data[yr].get(series, pd.DataFrame())
        if len(sub) == 0: continue
        p = sub[(sub['period'] == period) & sub[col].notna()].sort_values('date')
        if len(p) >= 2:
            out[yr] = p.reset_index(drop=True)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 1. Period Directional Bias
# ─────────────────────────────────────────────────────────────────────────────
def detect_period_bias(data, years, series, col='close_price', period='P1', min_hit=0.58):
    pdata = _period_data(data, years, series, col, period)
    if len(pdata) < 6: return []
    moves = {yr: df[col].iloc[-1] - df[col].iloc[0] for yr, df in pdata.items()}
    n = len(moves); results = []
    for direction in ['UP', 'DOWN']:
        yhit  = [yr for yr, v in moves.items() if (v > 0 if direction=='UP' else v < 0)]
        ymiss = [yr for yr in moves if yr not in yhit]
        hr = len(yhit) / n
        if hr < min_hit: continue
        avg_m = float(np.mean([moves[yr] for yr in yhit])) if yhit else 0
        all_m = list(moves.values())
        results.append(_f(
            series, period, 'Period Directional Bias',
            f'{direction} {len(yhit)}/{n}',
            f'{SERIES_LABELS.get(series,series)} finishes the {PERIOD_LABELS.get(period,period)} '
            f'{direction} — {len(yhit)}/{n} years ({hr:.0%}). '
            f'Average move when matching: {avg_m:+.0f} pts. '
            f'Full range: {min(all_m):+.0f} to {max(all_m):+.0f} pts.',
            len(yhit), n, yhit, ymiss,
            avg_move=avg_m, move_range=(min(all_m), max(all_m)),
        ))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 2. Consecutive N-Day Runs
# ─────────────────────────────────────────────────────────────────────────────
def detect_consecutive_runs(data, years, series, col='close_price', period='P1',
                             min_run=2, max_run=4, min_hit=0.58):
    pdata = _period_data(data, years, series, col, period)
    if len(pdata) < 6: return []
    max_bd = int(max(df['bd'].max() for df in pdata.values()))
    results = []

    for run_len in range(min_run, max_run + 1):
        for start_bd in range(1, max_bd + 1):
            end_bd = start_bd + run_len - 1
            if end_bd > max_bd: break
            for direction in ['UP', 'DOWN']:
                yhit, ymiss, moves_h, moves_a = [], [], [], []
                for yr, df in pdata.items():
                    win = df[(df['bd'] >= start_bd) & (df['bd'] <= end_bd)].reset_index(drop=True)
                    if len(win) < run_len: continue
                    vals = win[col].values
                    diffs = np.diff(vals[:run_len])
                    net = float(vals[min(run_len-1, len(vals)-1)] - vals[0])
                    moves_a.append(net)
                    match = bool((diffs > 0).all()) if direction=='UP' else bool((diffs < 0).all())
                    if match: yhit.append(yr); moves_h.append(net)
                    else:     ymiss.append(yr)
                n = len(yhit) + len(ymiss)
                if n < 6: continue
                hr = len(yhit) / n
                if hr < min_hit: continue
                avg_m = float(np.mean(moves_h)) if moves_h else 0
                mr = (float(min(moves_a)), float(max(moves_a))) if moves_a else None
                results.append(_f(
                    series, period,
                    f'Consecutive {run_len}-Day Run',
                    f'{direction}×{run_len} BD{start_bd}–{end_bd}',
                    f'{run_len} consecutive {direction} days (BD {start_bd}→{end_bd}) '
                    f'in {PERIOD_LABELS.get(period,period)} — '
                    f'{len(yhit)}/{n} years ({hr:.0%}). '
                    f'Avg cumulative move when pattern fires: {avg_m:+.0f} pts. '
                    f'Range across all years: {mr[0]:+.0f} to {mr[1]:+.0f}.' if mr else '',
                    len(yhit), n, yhit, ymiss,
                    avg_move=avg_m, move_range=mr, bd_start=start_bd, bd_end=end_bd,
                ))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 3. Day-Follows-Day (direction repetition)
# ─────────────────────────────────────────────────────────────────────────────
def detect_day_follows_day(data, years, series, col='close_price', period='P1', min_hit=0.65):
    pdata = _period_data(data, years, series, col, period)
    if len(pdata) < 6: return []
    max_bd = int(max(df['bd'].max() for df in pdata.values()))
    results = []
    for bd in range(2, max_bd):   # test bd and bd+1
        yhit, ymiss = [], []
        for yr, df in pdata.items():
            prev_row = df[df['bd'] == bd - 1][col].values
            curr_row = df[df['bd'] == bd][col].values
            next_row = df[df['bd'] == bd + 1][col].values
            if len(prev_row)==0 or len(curr_row)==0 or len(next_row)==0: continue
            chg_today = curr_row[0] - prev_row[0]
            chg_next  = next_row[0] - curr_row[0]
            if np.sign(chg_today) == np.sign(chg_next) and chg_today != 0:
                yhit.append(yr)
            else:
                ymiss.append(yr)
        n = len(yhit) + len(ymiss)
        if n < 6: continue
        hr = len(yhit) / n
        if hr < min_hit: continue
        results.append(_f(
            series, period, 'Day-Follows-Day',
            f'BD{bd}→BD{bd+1} same dir',
            f'{SERIES_LABELS.get(series,series)}: BD {bd+1} repeats BD {bd}\'s direction '
            f'— {len(yhit)}/{n} years ({hr:.0%}) in {PERIOD_LABELS.get(period,period)}. '
            f'If BD {bd} moves UP, BD {bd+1} tends to also move UP, and vice versa. '
            f'Useful for averaging in on day {bd+1} if day {bd} gave a clean entry.',
            len(yhit), n, yhit, ymiss, bd_start=bd, bd_end=bd+1,
            note=f'Trade: if BD {bd} direction is clear, add on BD {bd+1} open.'
        ))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 4. First-Day Predictor (BD1 predicts whole period)
# ─────────────────────────────────────────────────────────────────────────────
def detect_first_day_predictor(data, years, series, col='close_price', period='P2', min_hit=0.65):
    pdata = _period_data(data, years, series, col, period)
    if len(pdata) < 6: return []
    yhit, ymiss = [], []
    for yr, df in pdata.items():
        if len(df) < 3: continue
        first_day_chg = df[col].iloc[1] - df[col].iloc[0]
        period_chg    = df[col].iloc[-1] - df[col].iloc[0]
        if first_day_chg == 0 or period_chg == 0: continue
        if np.sign(first_day_chg) == np.sign(period_chg):
            yhit.append(yr)
        else:
            ymiss.append(yr)
    n = len(yhit) + len(ymiss)
    if n < 6: return []
    hr = len(yhit) / n
    if hr < min_hit: return []
    return [_f(
        series, period, 'First-Day Predictor',
        f'BD1 → period direction',
        f'{SERIES_LABELS.get(series,series)} in {PERIOD_LABELS.get(period,period)}: '
        f'the direction of BD 1 predicts the overall period direction — '
        f'{len(yhit)}/{n} years ({hr:.0%}). '
        f'Trade: enter in BD1\'s direction at end of BD1, hold for the period.',
        len(yhit), n, yhit, ymiss, bd_start=1, bd_end=None,
        note='Use BD1 close vs BD1 open (or prior close) as entry signal for the full window.'
    )]

# ─────────────────────────────────────────────────────────────────────────────
# 5. Pre-Roll Volume Spike (last P1 day)
# ─────────────────────────────────────────────────────────────────────────────
def detect_pre_roll_volume_spike(data, years, min_hit=0.55):
    yhit, ymiss, ratios = [], [], {}
    for yr in [y for y in years if y in data]:
        sub = data[yr].get('out', pd.DataFrame())
        if len(sub) == 0: continue
        p1 = sub[(sub['period']=='P1') & sub['Volume'].notna()].sort_values('date')
        if len(p1) < 5: continue
        last_vol = float(p1['Volume'].iloc[-1])
        avg_vol  = float(p1['Volume'].iloc[:-1].mean())
        if avg_vol == 0: continue
        ratios[yr] = last_vol / avg_vol
        (yhit if ratios[yr] > 1.30 else ymiss).append(yr)
    n = len(yhit) + len(ymiss)
    if n < 6: return []
    hr = len(yhit) / n
    if hr < min_hit: return []
    avg_r = float(np.mean([ratios[yr] for yr in yhit])) if yhit else 0
    return [_f(
        'out', 'P1→P2', 'Pre-Roll Volume Spike',
        f'Vol spike last P1 day',
        f'Outright volume on the LAST DAY of P1 (day before Goldman roll begins) is '
        f'at least 1.3× the P1 daily average — {len(yhit)}/{n} years ({hr:.0%}). '
        f'Average elevation on that day: {avg_r:.1f}× P1 average. '
        f'Signals pre-positioning by market participants ahead of the roll.',
        len(yhit), n, yhit, ymiss,
        vol_ctx=f'Last P1 day = avg {avg_r:.1f}× P1 daily volume',
        note='Monitor for volume buildup in final P1 days — often a leading indicator of roll pressure direction.'
    )]

# ─────────────────────────────────────────────────────────────────────────────
# 6. Volume Concentration in roll window
# ─────────────────────────────────────────────────────────────────────────────
def detect_volume_concentration(data, years, series='out', period='P2', min_hit=0.55):
    front, back, valid = [], [], []
    for yr in [y for y in years if y in data]:
        sub = data[yr].get(series, pd.DataFrame())
        if len(sub) == 0: continue
        p = sub[(sub['period']==period) & sub['Volume'].notna()].sort_values('date')
        if len(p) < 4: continue
        valid.append(yr)
        half = len(p) // 2
        f_avg = p['Volume'].iloc[:half].mean()
        b_avg = p['Volume'].iloc[half:].mean()
        if f_avg > b_avg * 1.25: front.append(yr)
        elif b_avg > f_avg * 1.25: back.append(yr)
    n = len(valid)
    if n < 6: return []
    results = []
    for yrs, label, detail in [
        (front, 'front-loaded (BD1–2 heaviest)',
         'Volume spikes at the START of the roll window and tails off. '
         'Index managers are aggressive on day 1–2 of Goldman roll.'),
        (back,  'back-loaded (BD4–5 heaviest)',
         'Volume builds toward the END of the roll window. '
         'Late roll concentration in BD 4–5.')
    ]:
        hr = len(yrs)/n
        if hr < min_hit: continue
        ymiss = [yr for yr in valid if yr not in yrs]
        results.append(_f(
            series, period, 'Volume Concentration',
            f'Vol {label.split("(")[0].strip()}',
            f'{SERIES_LABELS.get(series,series)} volume during {PERIOD_LABELS.get(period,period)} '
            f'is {label} — {len(yrs)}/{n} years ({hr:.0%}). {detail}',
            len(yrs), n, yrs, ymiss,
            vol_ctx=f'Roll volume is {label}'
        ))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 7. Peak Volume BD
# ─────────────────────────────────────────────────────────────────────────────
def detect_peak_volume_bd(data, years, series='out', period='P2', min_hit=0.40):
    bd_counts, valid = {}, []
    for yr in [y for y in years if y in data]:
        sub = data[yr].get(series, pd.DataFrame())
        if len(sub) == 0: continue
        p = sub[(sub['period']==period) & sub['Volume'].notna()].sort_values('date')
        if len(p) < 3: continue
        valid.append(yr)
        peak_bd = int(p.loc[p['Volume'].idxmax(), 'bd'])
        bd_counts[yr] = peak_bd
    if len(valid) < 6: return []
    cnt = Counter(bd_counts.values())
    results = []
    for bd, count in cnt.most_common(2):
        hr = count / len(valid)
        if hr < min_hit: continue
        yhit  = [yr for yr, b in bd_counts.items() if b == bd]
        ymiss = [yr for yr, b in bd_counts.items() if b != bd]
        results.append(_f(
            series, period, 'Peak Volume BD',
            f'Peak vol = BD{bd}',
            f'{SERIES_LABELS.get(series,series)} peak volume in {PERIOD_LABELS.get(period,period)} '
            f'falls on BD {bd} — {count}/{len(valid)} years ({hr:.0%}). '
            f'BD {bd} is the most active day of the roll window for this instrument.',
            count, len(valid), yhit, ymiss,
            bd_start=bd, bd_end=bd,
            vol_ctx=f'BD{bd} = peak volume day in {hr:.0%} of years',
            note=f'Expect maximum spread/price volatility on BD {bd} of the roll.'
        ))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 8. Rogers-Goldman Reversal (P1 direction reversed in P2)
# ─────────────────────────────────────────────────────────────────────────────
def detect_rogers_goldman_reversal(data, years, series, col='close_price', min_hit=0.58):
    yhit, ymiss, valid = [], [], []
    for yr in [y for y in years if y in data]:
        sub = data[yr].get(series, pd.DataFrame())
        if len(sub) == 0: continue
        p1 = sub[(sub['period']=='P1') & sub[col].notna()].sort_values('date')
        p2 = sub[(sub['period']=='P2') & sub[col].notna()].sort_values('date')
        if len(p1) < 3 or len(p2) < 3: continue
        valid.append(yr)
        p1_net = p1[col].iloc[-1] - p1[col].iloc[0]
        p2_net = p2[col].iloc[-1] - p2[col].iloc[0]
        if p1_net == 0 or p2_net == 0: continue
        (yhit if np.sign(p1_net) != np.sign(p2_net) else ymiss).append(yr)
    n = len(valid)
    if n < 6: return []
    hr = len(yhit) / n
    if hr < min_hit: return []
    return [_f(
        series, 'P1→P2', 'Rogers-Goldman Reversal',
        f'P1→P2 reversal',
        f'{SERIES_LABELS.get(series,series)}: Goldman Roll (P2) direction REVERSES '
        f'the P1 direction — {len(yhit)}/{n} years ({hr:.0%}). '
        f'If P1 was UP, P2 tends to be DOWN and vice versa. '
        f'Particularly noted in the Futures First research for cocoa outrights and spreads.',
        len(yhit), n, yhit, ymiss,
        note='Counter-trade setup: position OPPOSITE to P1 direction on first day of P2.'
    )]

# ─────────────────────────────────────────────────────────────────────────────
# 9. P2 → P3 Continuation
# ─────────────────────────────────────────────────────────────────────────────
def detect_p2_p3_continuation(data, years, series, col='close_price', min_hit=0.58):
    yhit, ymiss, valid = [], [], []
    for yr in [y for y in years if y in data]:
        sub = data[yr].get(series, pd.DataFrame())
        if len(sub) == 0: continue
        p2 = sub[(sub['period']=='P2') & sub[col].notna()].sort_values('date')
        p3 = sub[(sub['period']=='P3') & sub[col].notna()].sort_values('date')
        if len(p2) < 3 or len(p3) < 3: continue
        valid.append(yr)
        p2_net = p2[col].iloc[-1] - p2[col].iloc[0]
        p3_net = p3[col].iloc[-1] - p3[col].iloc[0]
        if p2_net == 0 or p3_net == 0: continue
        (yhit if np.sign(p2_net) == np.sign(p3_net) else ymiss).append(yr)
    n = len(valid)
    if n < 6: return []
    hr = len(yhit) / n
    if hr < min_hit: return []
    return [_f(
        series, 'P2→P3', 'P2-P3 Continuation',
        f'P2→P3 continues',
        f'{SERIES_LABELS.get(series,series)}: P3 (Post-Roll) continues the Goldman Roll direction '
        f'— {len(yhit)}/{n} years ({hr:.0%}). '
        f'The direction established during the Goldman roll tends to persist into expiry.',
        len(yhit), n, yhit, ymiss,
        note='If Goldman roll direction is clear by BD3, consider holding into P3.'
    )]

# ─────────────────────────────────────────────────────────────────────────────
# 10. OI Direction Pattern
# ─────────────────────────────────────────────────────────────────────────────
def detect_oi_pattern(data, years, period='P2', min_hit=0.60):
    results = []
    for direction in ['FALLING', 'RISING']:
        yhit, ymiss, valid, changes = [], [], [], []
        for yr in [y for y in years if y in data]:
            sub = data[yr].get('out', pd.DataFrame())
            if len(sub) == 0: continue
            p = sub[(sub['period']==period) & sub['Open Interest'].notna()].sort_values('date')
            if len(p) < 3: continue
            valid.append(yr)
            net = float(p['Open Interest'].iloc[-1] - p['Open Interest'].iloc[0])
            changes.append(net)
            (yhit if (net < 0 if direction=='FALLING' else net > 0) else ymiss).append(yr)
        n = len(valid)
        if n < 6: continue
        hr = len(yhit) / n
        if hr < min_hit: continue
        avg_chg = float(np.mean([changes[i] for i, yr in enumerate(valid) if yr in yhit]))
        context = ('Index selling front-month drives OI lower — normal Goldman roll signature.'
                   if direction=='FALLING' else 'New positioning being built in this period.')
        results.append(_f(
            'out', period, 'OI Direction Pattern',
            f'OI {direction} in {period}',
            f'Open Interest is {direction.lower()} during {PERIOD_LABELS.get(period,period)} '
            f'— {len(yhit)}/{n} years ({hr:.0%}). '
            f'Average OI change: {avg_chg:+,.0f} contracts. {context}',
            len(yhit), n, yhit, ymiss,
            oi_ctx=f'Avg OI Δ: {avg_chg:+,.0f} contracts in matching years'
        ))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 11. OI-Price Classification (new longs / new shorts / covering / liquidation)
# ─────────────────────────────────────────────────────────────────────────────
def detect_oi_price_classification(data, years, series='out', period='P2'):
    rows = []
    for yr in [y for y in years if y in data]:
        sub = data[yr].get(series, pd.DataFrame())
        if len(sub) == 0: continue
        p = sub[(sub['period']==period)].copy()
        if 'oi_chg' not in p.columns or p['oi_chg'].isna().all(): continue
        p = p[p['oi_chg'].notna() & p['px_chg'].notna()].sort_values('date')
        p['year'] = yr
        rows.append(p[['bd','year','close_price','px_chg','oi_chg','Volume']])
    if not rows: return []
    combined = pd.concat(rows, ignore_index=True)
    if len(combined) < 10: return []

    cls_map = {
        (True,  True):  'New Longs',
        (False, True):  'New Shorts',
        (True,  False): 'Short Covering',
        (False, False): 'Long Liquidation',
    }
    cls_detail = {
        'New Longs':        'Price ↑ + OI ↑ — buyers entering, bullish conviction.',
        'New Shorts':       'Price ↓ + OI ↑ — sellers entering, bearish conviction.',
        'Short Covering':   'Price ↑ + OI ↓ — shorts being forced out, move may be exhaustive.',
        'Long Liquidation': 'Price ↓ + OI ↓ — longs exiting, move may be exhaustive.',
    }
    combined['cls'] = combined.apply(
        lambda r: cls_map[(r['px_chg'] > 0, r['oi_chg'] > 0)], axis=1)

    results = []
    for bd in sorted(combined['bd'].dropna().unique()):
        bd = int(bd)
        sub_bd = combined[combined['bd'] == bd]
        if len(sub_bd) < 5: continue
        cnt = sub_bd['cls'].value_counts()
        dom_cls = cnt.index[0]; dom_cnt = cnt.iloc[0]; total = len(sub_bd)
        pct = dom_cnt / total
        if pct < 0.50: continue
        yhit  = sorted(sub_bd[sub_bd['cls']==dom_cls]['year'].unique().tolist())
        ymiss = sorted(sub_bd[sub_bd['cls']!=dom_cls]['year'].unique().tolist())
        results.append(_f(
            series, period, 'OI-Price Classification',
            f'BD{bd}: {dom_cls}',
            f'On BD {bd} of {PERIOD_LABELS.get(period,period)}: '
            f'dominant pattern is {dom_cls} — {dom_cnt}/{total} years ({pct:.0%}). '
            f'{cls_detail[dom_cls]}',
            dom_cnt, total, yhit, ymiss,
            bd_start=bd, bd_end=bd,
            oi_ctx=f'BD{bd}: {dom_cls} dominant ({pct:.0%})'
        ))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 12. Momentum Statistics (informational card)
# ─────────────────────────────────────────────────────────────────────────────
def compute_momentum_stats(data, years, series, col='close_price', period='P1'):
    pdata = _period_data(data, years, series, col, period)
    if len(pdata) < 4: return []
    moves = [float(df[col].iloc[-1] - df[col].iloc[0]) for df in pdata.values()]
    ranges = []
    for df in pdata.values():
        if 'High' in df.columns and 'Low' in df.columns:
            h = df['High'].dropna(); l = df['Low'].dropna()
            if len(h) and len(l): ranges.append(float(h.max() - l.min()))
    avg_m  = float(np.mean(moves));  std_m = float(np.std(moves))
    p25    = float(np.percentile(moves, 25)); p75 = float(np.percentile(moves, 75))
    up_n   = sum(1 for m in moves if m > 0); dn_n = len(moves) - up_n
    avg_r  = float(np.mean(ranges)) if ranges else None
    desc = (
        f'{SERIES_LABELS.get(series,series)} · {PERIOD_LABELS.get(period,period)}: '
        f'Average net move {avg_m:+.0f} pts (σ={std_m:.0f}). '
        f'IQR [{p25:+.0f}, {p75:+.0f}] pts. '
        f'Up {up_n}/{len(moves)} years, Down {dn_n}/{len(moves)} years.'
    )
    if avg_r: desc += f' Avg intraperiod range (H−L): {avg_r:.0f} pts.'
    yhit_list = [yr for yr, df in pdata.items() if df[col].iloc[-1]-df[col].iloc[0] > 0]
    ymiss_list = [yr for yr in pdata if yr not in yhit_list]
    dominant_n = max(up_n, dn_n)
    return [_f(
        series, period, 'Momentum Statistics',
        f'Stats: avg {avg_m:+.0f} pts',
        desc,
        dominant_n, len(moves), yhit_list, ymiss_list,
        avg_move=avg_m, move_range=(float(min(moves)), float(max(moves))),
    )]

# ─────────────────────────────────────────────────────────────────────────────
# 13. Similar year finder (no finding card, returns dict)
# ─────────────────────────────────────────────────────────────────────────────
def find_similar_years(data, years, ref_year=None, n_similar=3):
    if ref_year is None: ref_year = max(years)
    if ref_year not in data: return {}
    ref_out = data[ref_year].get('out', pd.DataFrame())
    if len(ref_out) == 0: return {}
    ref_p1 = ref_out[ref_out['period']=='P1'].sort_values('date')
    if len(ref_p1) == 0: return {}
    ref_price = float(ref_p1['close_price'].dropna().iloc[0]) if ref_p1['close_price'].notna().any() else None
    ref_oi    = float(ref_p1['Open Interest'].dropna().iloc[0]) if ref_p1['Open Interest'].notna().any() else None
    if ref_price is None: return {}
    scores = {}
    for yr in years:
        if yr == ref_year: continue
        sub = data[yr].get('out', pd.DataFrame())
        if len(sub) == 0: continue
        p1 = sub[sub['period']=='P1'].sort_values('date')
        if len(p1) == 0: continue
        if p1['close_price'].isna().all(): continue
        price_diff = abs(float(p1['close_price'].dropna().iloc[0]) - ref_price) / ref_price
        oi_diff = 0.0
        if ref_oi and p1['Open Interest'].notna().any():
            oi_diff = abs(float(p1['Open Interest'].dropna().iloc[0]) - ref_oi) / ref_oi
        scores[yr] = price_diff * 0.6 + oi_diff * 0.4
    return dict(sorted(scores.items(), key=lambda x: x[1])[:n_similar])

# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────
def run_all_patterns(data, years, min_hit=0.58):
    all_f = []
    for series in ['out', 'hk', 'fly']:
        for period in ['P1', 'P2', 'P3']:
            all_f += detect_period_bias(data, years, series, 'close_price', period, min_hit)
            all_f += detect_consecutive_runs(data, years, series, 'close_price', period, 2, 4, min_hit)
            all_f += detect_day_follows_day(data, years, series, 'close_price', period, max(min_hit, 0.62))
            all_f += compute_momentum_stats(data, years, series, 'close_price', period)
            if period == 'P2':
                all_f += detect_first_day_predictor(data, years, series, 'close_price', period, max(min_hit, 0.62))
                all_f += detect_oi_price_classification(data, years, series, period)
                all_f += detect_peak_volume_bd(data, years, series, period, 0.40)
                all_f += detect_volume_concentration(data, years, series, period, min_hit)
        all_f += detect_rogers_goldman_reversal(data, years, series, 'close_price', min_hit)
        all_f += detect_p2_p3_continuation(data, years, series, 'close_price', min_hit)
    all_f += detect_pre_roll_volume_spike(data, years, min_hit)
    for period in ['P1', 'P2', 'P3']:
        all_f += detect_oi_pattern(data, years, period, 0.60)
    # Deduplicate by tag
    seen, unique = set(), []
    for f in all_f:
        k = (f['series'], f['period'], f['tag'])
        if k not in seen:
            seen.add(k)
            unique.append(f)
    co = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    unique.sort(key=lambda x: (co[x['confidence']], -x['hit_rate']))
    return unique
