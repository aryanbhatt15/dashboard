"""
CC March (CC H) — Analysis Dashboard  v2
=========================================
8 Tabs: Overview | Outright | H-K Spread | HKN Fly |
        Roll Window | Week Analysis | BD Calendar | Pattern Engine
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data_engine import (
    load_all, directional_bias, consistent_pattern_days, seasonal_scorecard,
    oi_cliff, range_ratio, vol_oi_patterns, spread_move_vs_roll,
    streak_analysis, return_dist, autocorr_stats, vol_anomaly_stats,
    cross_corr, week_stats, bd_to_date_map, current_bd_info, count_bd
)

# ── Load ──────────────────────────────────────────────────────────────────────
DATA, YEARS = load_all()
BD_MAP = bd_to_date_map(DATA, YEARS)
YEAR_OPTS = [{'label': str(y), 'value': y} for y in YEARS]
from pattern_engine_v2 import run_all_patterns, find_similar_years
CACHED_PATTERNS = run_all_patterns(DATA, YEARS, min_hit=0.58)


# ── Theme ─────────────────────────────────────────────────────────────────────
BG      = '#0d1117'
PAPER   = '#161b22'
CARD_BG = '#1c2128'
BORDER  = '#30363d'
TEXT    = '#e6edf3'
MUTED   = '#8b949e'
P_COL   = {'P1': '#388bfd', 'P2': '#f0883e', 'P3': '#da3633'}
GS_COL  = '#f0883e'
RG_COL  = '#3fb950'
YR_COLS = ['#58a6ff','#3fb950','#f0883e','#da3633','#bc8cff',
           '#79c0ff','#56d364','#ffa657','#ff7b72','#d2a8ff',
           '#a5d6ff','#7ee787']

def base_layout(**kw):
    d = dict(paper_bgcolor=PAPER, plot_bgcolor=BG,
             font=dict(color=TEXT, family='Inter,sans-serif', size=11),
             xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER,
                        rangebreaks=[dict(bounds=['sat','mon'])]),
             yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
             margin=dict(l=50, r=20, t=36, b=36),
             legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=BORDER, font_size=10),
             hovermode='x unified')
    d.update(kw)
    return d

def apply_base(fig, rows=1, **kw):
    layout = base_layout(**kw)
    fig.update_layout(**layout)
    if rows > 1:
        for i in range(1, rows+1):
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat","mon"])],
                             gridcolor=BORDER, row=i)
            fig.update_yaxes(gridcolor=BORDER, row=i)
    else:
        fig.update_xaxes(gridcolor=BORDER)
        fig.update_yaxes(gridcolor=BORDER)
    return fig

CARD = {'background': CARD_BG, 'borderRadius': '8px', 'padding': '16px',
        'marginBottom': '14px', 'border': f'1px solid {BORDER}'}
LABEL = {'color': MUTED, 'fontSize': '11px', 'textTransform': 'uppercase',
         'letterSpacing': '.05em', 'marginBottom': '4px'}

def add_period_bands(fig, bounds, rows=1):
    for p, col in P_COL.items():
        sk = 'p1_start' if p=='P1' else 'p2_start' if p=='P2' else 'p3_start'
        ek = 'p1_end'   if p=='P1' else 'p2_end'   if p=='P2' else 'mar_ltd'
        for r in range(1, rows+1):
            fig.add_vrect(x0=str(bounds[sk]), x1=str(bounds[ek]),
                          fillcolor=col, opacity=0.06, line_width=0,
                          row=r, col=1)
    return fig

def add_roll_lines(fig, bounds, rows=1):
    for r in range(1, rows+1):
        fig.add_vline(x=str(bounds['p2_start']), line_dash='dash',
                      line_color=GS_COL, line_width=1, row=r, col=1,
                      annotation_text='GS', annotation_font_size=9,
                      annotation_font_color=GS_COL)
        fig.add_vline(x=str(bounds['rogers_out_start']), line_dash='dot',
                      line_color=RG_COL, line_width=1, row=r, col=1,
                      annotation_text='Rog', annotation_font_size=9,
                      annotation_font_color=RG_COL)
    return fig

# ── Chart builders ────────────────────────────────────────────────────────────

def chart_ohlc_oi_vol(year):
    """3-pane: OHLC candlestick | OI | Volume — single year."""
    sub = DATA[year]['out']
    if len(sub) == 0: return go.Figure()
    b = DATA[year]['bounds']
    sub = sub[sub['close_price'].notna()].sort_values('Timestamp')

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=['Price (OHLC / Last)',
                                        'Open Interest & Daily Change',
                                        'Volume'],
                        row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.04)

    # Candlestick where H/L available; line otherwise
    ohlc = sub[sub['High'].notna()]
    line = sub[sub['High'].isna()]

    if len(ohlc):
        fig.add_trace(go.Candlestick(
            x=ohlc['Timestamp'], open=ohlc['Open'], high=ohlc['High'],
            low=ohlc['Low'], close=ohlc['Last'],
            name='OHLC', increasing_line_color='#3fb950',
            decreasing_line_color='#da3633'), row=1, col=1)
    if len(line):
        fig.add_trace(go.Scatter(
            x=line['Timestamp'], y=line['close_price'], mode='markers',
            name='Close (no OHLC)', marker=dict(color='#8b949e', size=3)), row=1, col=1)

    oi = sub[sub['Open Interest'].notna()]
    fig.add_trace(go.Scatter(x=oi['Timestamp'], y=oi['Open Interest'],
        fill='tozeroy', fillcolor='rgba(56,139,253,0.15)',
        line=dict(color='#388bfd', width=1.5), name='OI'), row=2, col=1)
    oi2 = oi[oi['oi_chg'].notna()]
    bar_col = ['#3fb950' if v >= 0 else '#da3633' for v in oi2['oi_chg']]
    fig.add_trace(go.Bar(x=oi2['Timestamp'], y=oi2['oi_chg'],
        name='OI Δ', marker_color=bar_col, opacity=0.6), row=2, col=1)

    vol = sub[sub['Volume'].notna()]
    fig.add_trace(go.Bar(x=vol['Timestamp'], y=vol['Volume'],
        name='Volume', marker_color='rgba(240,136,62,0.65)'), row=3, col=1)

    add_period_bands(fig, b, rows=3)
    add_roll_lines(fig, b, rows=3)
    apply_base(fig, rows=3, title=f'CC H {year} — Price | OI | Volume',
               height=620, xaxis_rangeslider_visible=False)
    return fig

def chart_atr(years_sel, period='P1'):
    """ATR (daily range H-L) per BD — multi-year."""
    fig = go.Figure()
    for i, yr in enumerate(years_sel):
        sub = DATA[yr]['out']
        p = sub[(sub['period']==period) & sub['daily_range'].notna()].sort_values('date')
        if len(p) == 0: continue
        fig.add_trace(go.Scatter(x=p['bd'], y=p['daily_range'],
            mode='lines+markers', name=str(yr),
            line=dict(color=YR_COLS[i % len(YR_COLS)], width=1.5),
            marker=dict(size=4),
            hovertemplate=f'{yr} BD%{{x}}: %{{y:.0f}}<extra></extra>'))
    # avg line
    rows = []
    for yr in years_sel:
        sub = DATA[yr]['out']
        p = sub[(sub['period']==period) & sub['daily_range'].notna()][['bd','daily_range']]
        rows.append(p)
    if rows:
        combined = pd.concat(rows).groupby('bd')['daily_range'].mean().reset_index()
        fig.add_trace(go.Scatter(x=combined['bd'], y=combined['daily_range'],
            mode='lines', name='Mean', line=dict(color='white', width=2.5, dash='dot')))
    apply_base(fig, title=f'Outright Daily Range (H-L) — {period}',
               xaxis_title='BD in Period', yaxis_title='Points', height=320)
    return fig

def chart_oi_by_bd(years_sel, series='out', period='P1'):
    """OI level per BD — multi-year."""
    fig = go.Figure()
    for i, yr in enumerate(years_sel):
        sub = DATA[yr].get(series, pd.DataFrame())
        if len(sub) == 0: continue
        p = sub[(sub['period']==period) & sub['Open Interest'].notna()].sort_values('date')
        if len(p) == 0: continue
        fig.add_trace(go.Scatter(x=p['bd'], y=p['Open Interest'],
            mode='lines+markers', name=str(yr),
            line=dict(color=YR_COLS[i % len(YR_COLS)], width=1.5),
            marker=dict(size=3),
            hovertemplate=f'{yr} BD%{{x}}: %{{y:,.0f}}<extra></extra>'))
    rows = [DATA[yr].get(series,pd.DataFrame()) for yr in years_sel]
    rows = [r[(r['period']==period) & r['Open Interest'].notna()][['bd','Open Interest']] for r in rows if len(r)]
    if rows:
        m = pd.concat(rows).groupby('bd')['Open Interest'].mean().reset_index()
        fig.add_trace(go.Scatter(x=m['bd'], y=m['Open Interest'],
            mode='lines', name='Mean', line=dict(color='white', width=2.5, dash='dot')))
    apply_base(fig, title=f'Open Interest by BD — {period}',
               xaxis_title='BD in Period', yaxis_title='OI', height=300)
    return fig

def chart_vol_by_bd(years_sel, series='out', period='P1'):
    """Volume per BD — multi-year."""
    fig = go.Figure()
    for i, yr in enumerate(years_sel):
        sub = DATA[yr].get(series, pd.DataFrame())
        if len(sub) == 0: continue
        p = sub[(sub['period']==period) & sub['Volume'].notna()].sort_values('date')
        if len(p) == 0: continue
        fig.add_trace(go.Bar(x=p['bd'], y=p['Volume'],
            name=str(yr), marker_color=YR_COLS[i % len(YR_COLS)],
            opacity=0.65))
    apply_base(fig, title=f'Volume by BD — {period}',
               xaxis_title='BD in Period', yaxis_title='Volume',
               barmode='group', height=300)
    return fig

def chart_spread_detail(year, series='hk'):
    """Spread: Last price (bar chart H-L range, close as dot) + Volume."""
    sub = DATA[year].get(series, pd.DataFrame())
    if len(sub) == 0: return go.Figure()
    b = DATA[year]['bounds']
    sub = sub[sub['close_price'].notna()].sort_values('Timestamp')
    label = 'H-K Spread' if series == 'hk' else 'K-N Spread'

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=[f'{label} — Last Price & Range', 'Volume'],
                        row_heights=[0.7, 0.3], vertical_spacing=0.06)

    # Show H-L range where available, else just close dot
    ohlc = sub[sub['High'].notna()]
    if len(ohlc):
        fig.add_trace(go.Bar(x=ohlc['Timestamp'],
            y=ohlc['High'] - ohlc['Low'],
            base=ohlc['Low'],
            name='H-L Range', marker_color='rgba(56,139,253,0.25)',
            width=24*3600*1000), row=1, col=1)
        fig.add_trace(go.Scatter(x=ohlc['Timestamp'], y=ohlc['Last'],
            mode='markers', name='Last',
            marker=dict(color='#388bfd', size=6, symbol='circle')), row=1, col=1)

    no_ohlc = sub[sub['High'].isna()]
    if len(no_ohlc):
        fig.add_trace(go.Scatter(x=no_ohlc['Timestamp'], y=no_ohlc['close_price'],
            mode='markers', name='Settlement',
            marker=dict(color='#8b949e', size=4, symbol='x')), row=1, col=1)

    fig.add_hline(y=0, line_dash='dash', line_color='#8b949e', row=1, col=1)

    vol = sub[sub['Volume'].notna()]
    fig.add_trace(go.Bar(x=vol['Timestamp'], y=vol['Volume'],
        name='Volume', marker_color='rgba(240,136,62,0.6)'), row=2, col=1)

    add_period_bands(fig, b, rows=2)
    add_roll_lines(fig, b, rows=2)
    apply_base(fig, rows=2, title=f'{label} — {year}', height=440)
    return fig

def chart_spread_by_bd(years_sel, series='hk', period='P1'):
    """Spread close_price per BD — multi-year."""
    label = 'H-K Spread' if series == 'hk' else 'K-N Spread' if series == 'kn' else 'HKN Fly'
    fig = go.Figure()
    for i, yr in enumerate(years_sel):
        sub = DATA[yr].get(series, pd.DataFrame())
        if len(sub) == 0: continue
        p = sub[(sub['period']==period) & sub['close_price'].notna()].sort_values('date')
        if len(p) == 0: continue
        fig.add_trace(go.Scatter(x=p['bd'], y=p['close_price'],
            mode='lines+markers', name=str(yr),
            line=dict(color=YR_COLS[i % len(YR_COLS)], width=1.8),
            marker=dict(size=4),
            hovertemplate=f'{yr} BD%{{x}}: %{{y:.0f}}<extra></extra>'))
    rows = [DATA[yr].get(series,pd.DataFrame()) for yr in years_sel]
    rows = [r[(r['period']==period) & r['close_price'].notna()][['bd','close_price']] for r in rows if len(r)]
    if rows:
        m = pd.concat(rows).groupby('bd')['close_price'].mean().reset_index()
        fig.add_trace(go.Scatter(x=m['bd'], y=m['close_price'],
            mode='lines', name='Mean', line=dict(color='white', width=2.5, dash='dot')))
    fig.add_hline(y=0, line_dash='dash', line_color='#8b949e')
    apply_base(fig, title=f'{label} by BD — {period}',
               xaxis_title='BD in Period', yaxis_title='Points', height=320)
    return fig

def chart_fly_detail(year):
    sub = DATA[year].get('fly', pd.DataFrame())
    if len(sub) == 0: return go.Figure()
    b = DATA[year]['bounds']

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=['HKN Fly (H − 2K + N)', 'Fly Volume (avg)'],
                        row_heights=[0.7, 0.3], vertical_spacing=0.06)

    sub = sub[sub['close_price'].notna()].sort_values('Timestamp')
    fig.add_trace(go.Scatter(x=sub['Timestamp'], y=sub['close_price'],
        mode='lines+markers', name='Fly',
        line=dict(color='#bc8cff', width=2), marker=dict(size=3)), row=1, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='#8b949e', row=1, col=1)

    vol = sub[sub['Volume'].notna()]
    fig.add_trace(go.Bar(x=vol['Timestamp'], y=vol['Volume'],
        name='Volume', marker_color='rgba(188,140,255,0.5)'), row=2, col=1)

    add_period_bands(fig, b, rows=2)
    add_roll_lines(fig, b, rows=2)
    apply_base(fig, rows=2, title=f'HKN Fly — {year}', height=420)
    return fig

def chart_hit_rate_bar(years_sel, series='out', period='P1'):
    vop = vol_oi_patterns(DATA, years_sel, series, period)
    if len(vop) == 0: return go.Figure()
    clrs = ['#3fb950' if v >= 0.5 else '#da3633' for v in vop['hit_rate_up']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=vop['bd'], y=vop['hit_rate_up'],
        marker_color=clrs,
        text=[f'{v:.0%}' for v in vop['hit_rate_up']], textposition='outside'))
    fig.add_hline(y=0.5, line_dash='dash', line_color='#8b949e')
    fig.add_hline(y=0.7, line_dash='dot', line_color='#f0883e',
                  annotation_text='70%', annotation_font_color='#f0883e')
    apply_base(fig, title=f'Hit Rate UP per BD — {series.upper()} {period}',
               yaxis_range=[0, 1.1], xaxis_title='BD', yaxis_title='Fraction UP', height=260)
    return fig

def chart_net_move_bar(years_sel, series='out', period='P1'):
    df, _ = directional_bias(DATA, years_sel, series, 'close_price', period)
    if len(df) == 0: return go.Figure()
    clrs = ['#3fb950' if d == 'UP' else '#da3633' for d in df['direction']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['year'], y=df['net_move'],
        marker_color=clrs,
        text=[f'{v:+.0f}' for v in df['net_move']], textposition='outside'))
    fig.add_hline(y=0, line_color='#8b949e')
    label = {'out': 'Outright', 'hk': 'H-K Spread', 'kn': 'K-N Spread', 'fly': 'HKN Fly'}.get(series, series)
    apply_base(fig, title=f'{label} Net Move {period} — Year by Year',
               xaxis_title='Year', yaxis_title='Points', height=260)
    return fig

def chart_oi_decay(years_sel):
    from data_engine import oi_cliff
    cliff = oi_cliff(DATA, YEARS)
    fig = go.Figure()
    for i, yr in enumerate(years_sel):
        sub = DATA[yr]['out']
        sub = sub[sub['Open Interest'].notna()].sort_values('bd_to_ltd', ascending=False)
        fig.add_trace(go.Scatter(x=sub['bd_to_ltd'], y=sub['Open Interest'],
            mode='lines', name=str(yr),
            line=dict(color=YR_COLS[i % len(YR_COLS)], width=1.5),
            hovertemplate=f'{yr} %{{x}}bd to LTD: %{{y:,.0f}}<extra></extra>'))
    if len(cliff):
        med = cliff['bd_before_ltd'].median()
        fig.add_vline(x=med, line_dash='dash', line_color='yellow',
                      annotation_text=f'Median cliff {med:.0f}bd',
                      annotation_font_color='yellow')
    apply_base(fig, title='OI Decay — BD before LTD (all years)',
               xaxis_title='BD before LTD', yaxis_title='OI', height=320)
    return fig

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'CC March Analysis v2'

app.index_string = '''<!DOCTYPE html><html><head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  *{box-sizing:border-box}
  body{margin:0;background:#0d1117;font-family:Inter,-apple-system,sans-serif}
  .Select-control,.Select-menu-outer{background:#1c2128!important;border-color:#30363d!important;color:#e6edf3!important}
  .Select-value-label,.Select-placeholder{color:#e6edf3!important}
  .Select-option{background:#1c2128!important;color:#e6edf3!important}
  .Select-option:hover,.Select-option.is-focused{background:#30363d!important}
  ::-webkit-scrollbar{width:5px;background:#0d1117}
  ::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}
  input[type=radio]{accent-color:#388bfd}
  .dash-tab--selected{border-top:2px solid #388bfd!important}
</style>
</head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''

def ctrl_label(txt):
    return html.Div(txt, style=LABEL)

def metric_pill(label, val, sub='', color=TEXT):
    return html.Div([
        html.Div(label, style={**LABEL, 'marginBottom':'2px'}),
        html.Div(val,   style={'color': color, 'fontSize':'18px', 'fontWeight':'500'}),
        html.Div(sub,   style={'color': MUTED, 'fontSize':'11px'}),
    ], style={**CARD, 'padding':'10px 14px', 'minWidth':'120px', 'flex':'0 0 auto'})

# ── LAYOUT ────────────────────────────────────────────────────────────────────
app.layout = html.Div(style={'background':BG,'minHeight':'100vh','color':TEXT}, children=[

    # Header
    html.Div([
        html.H1('CC March (H) — Analysis Dashboard',
                style={'margin':'0 0 3px','fontSize':'18px','fontWeight':'500'}),
        html.Div([
            html.Span('NY Cocoa Futures · 2014–2025 · ', style={'color':MUTED}),
            html.Span('P1', style={'color':P_COL['P1'],'fontWeight':'500'}),
            html.Span(' Post-Nov LTD → Pre-Roll  ', style={'color':MUTED}),
            html.Span('P2', style={'color':P_COL['P2'],'fontWeight':'500'}),
            html.Span(' Goldman Roll (5th–9th BD Feb)  ', style={'color':MUTED}),
            html.Span('P3', style={'color':P_COL['P3'],'fontWeight':'500'}),
            html.Span(' Post-Roll → March LTD', style={'color':MUTED}),
        ], style={'fontSize':'12px'}),
    ], style={'padding':'16px 24px 10px','borderBottom':f'1px solid {BORDER}'}),

    # Controls bar
    html.Div([
        html.Div([
            ctrl_label('Single year'),
            dcc.Dropdown(id='yr-single', options=YEAR_OPTS, value=2022,
                         clearable=False, style={'width':'110px'}),
        ], style={'marginRight':'20px'}),
        html.Div([
            ctrl_label('Years for overlays / patterns'),
            dcc.Dropdown(id='yr-multi', options=YEAR_OPTS, value=YEARS,
                         multi=True, style={'width':'480px'}),
        ], style={'marginRight':'20px'}),
        html.Div([
            ctrl_label('Period'),
            dcc.RadioItems(id='period-sel',
                options=[{'label':p,'value':p} for p in ['P1','P2','P3']],
                value='P1', inline=True,
                labelStyle={'marginRight':'12px','fontSize':'13px'}),
        ], style={'marginRight':'20px'}),
        html.Div([
            ctrl_label('Series (overlays)'),
            dcc.RadioItems(id='series-sel',
                options=[{'label':l,'value':v} for l,v in
                         [('Outright','out'),('H-K Spread','hk'),('HKN Fly','fly')]],
                value='out', inline=True,
                labelStyle={'marginRight':'12px','fontSize':'13px'}),
        ]),
    ], style={'padding':'10px 24px','display':'flex','alignItems':'flex-end',
              'borderBottom':f'1px solid {BORDER}','gap':'4px','flexWrap':'wrap'}),

    # Tabs
    dcc.Tabs(id='tabs', value='tab-overview',
             colors={'border':BORDER,'primary':'#388bfd','background':PAPER},
             style={'background':BG},
    children=[
        dcc.Tab(label='📅 Overview',     value='tab-overview',
                style={'background':PAPER,'color':MUTED},
                selected_style={'background':BG,'color':TEXT}),
        dcc.Tab(label='📈 Outright',     value='tab-out',
                style={'background':PAPER,'color':MUTED},
                selected_style={'background':BG,'color':TEXT}),
        dcc.Tab(label='↔ H-K Spread',   value='tab-hk',
                style={'background':PAPER,'color':MUTED},
                selected_style={'background':BG,'color':TEXT}),
        dcc.Tab(label='🦋 HKN Fly',      value='tab-fly',
                style={'background':PAPER,'color':MUTED},
                selected_style={'background':BG,'color':TEXT}),
        dcc.Tab(label='🔄 Roll Window',  value='tab-roll',
                style={'background':PAPER,'color':MUTED},
                selected_style={'background':BG,'color':TEXT}),
        dcc.Tab(label='📆 Week Analysis',value='tab-week',
                style={'background':PAPER,'color':MUTED},
                selected_style={'background':BG,'color':TEXT}),
        dcc.Tab(label='📋 BD Calendar',  value='tab-cal',
                style={'background':PAPER,'color':MUTED},
                selected_style={'background':BG,'color':TEXT}),
        dcc.Tab(label='🔍 Pattern Engine',value='tab-pat',
                style={'background':PAPER,'color':MUTED},
                selected_style={'background':BG,'color':TEXT}),
    ]),
    html.Div(id='tab-content', style={'padding':'18px 24px'}),
])

# ── Shared table style ────────────────────────────────────────────────────────
def make_table(df, conditional=None, height='300px'):
    cond = conditional or []
    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name':c,'id':c} for c in df.columns],
        style_table={'overflowX':'auto','maxHeight':height,'overflowY':'auto'},
        style_header={'backgroundColor':BORDER,'color':TEXT,'fontWeight':'500','fontSize':'11px'},
        style_cell={'backgroundColor':PAPER,'color':TEXT,'fontSize':'11px',
                    'border':f'1px solid {BORDER}','padding':'5px 9px'},
        style_data_conditional=cond,
        sort_action='native', filter_action='native',
        page_size=50,
    )

UP_STYLE   = [{'if':{'filter_query':'{Direction} = "▲"','column_id':'Direction'},'color':'#3fb950','fontWeight':'bold'}]
DOWN_STYLE = [{'if':{'filter_query':'{Direction} = "▼"','column_id':'Direction'},'color':'#da3633','fontWeight':'bold'}]

# ── CALLBACKS ─────────────────────────────────────────────────────────────────

@app.callback(Output('tab-content','children'),
              Input('tabs','value'),
              Input('yr-single','value'),
              Input('yr-multi','value'),
              Input('period-sel','value'),
              Input('series-sel','value'))
def render_tab(tab, year, years_sel, period, series):
    years_sel = years_sel or YEARS
    year      = year or 2022

    # ── OVERVIEW ──────────────────────────────────────────────────────────────
    if tab == 'tab-overview':
        b = DATA[year]['bounds']

        # current BD info
        ci = current_bd_info(DATA, years_sel)
        cur_pos = ci['positions'][0] if ci['positions'] else {}

        # boundary table
        brows = []
        for yr in sorted(years_sel):
            bb = DATA[yr]['bounds']
            brows.append({'Year':yr,
                'Nov Last BD':str(bb['nov_last_bd']),
                'P1 Start':str(bb['p1_start']),'P1 End':str(bb['p1_end']),'P1 BDs':bb['p1_bd'],
                'Goldman Start':str(bb['p2_start']),'Goldman End':str(bb['p2_end']),
                'Rogers OUT Start':str(bb['rogers_out_start']),'Rogers OUT End':str(bb['rogers_out_end']),
                'P3 Start':str(bb['p3_start']),'Mar LTD':str(bb['mar_ltd']),'P3 BDs':bb['p3_bd']})
        bdf = pd.DataFrame(brows)

        # metric pills
        df_bias, st_bias = directional_bias(DATA, years_sel, 'out', 'close_price', 'P1')
        df_hk,   st_hk   = directional_bias(DATA, years_sel, 'hk',  'close_price', 'P2')
        cliff            = oi_cliff(DATA, years_sel)

        def pill(lbl, val, sub='', col=TEXT):
            return metric_pill(lbl, val, sub, col)

        up_col  = '#3fb950'; dn_col = '#da3633'
        bias_col = up_col if st_bias.get('direction') == 'UP' else dn_col
        hk_col   = up_col if st_hk.get('direction')  == 'UP' else dn_col

        pills = html.Div([
            pill('Outright P1 Bias',
                 f'{st_bias.get("direction","?")} {st_bias.get("hit_rate",0):.0%}',
                 f'avg {st_bias.get("avg_move",0):+.0f} pts, p={st_bias.get("pval",1):.2f}',
                 bias_col),
            pill('H-K Spread P2 Bias',
                 f'{st_hk.get("direction","?")} {st_hk.get("hit_rate",0):.0%}',
                 f'avg {st_hk.get("avg_move",0):+.0f} pts',
                 hk_col),
            pill('OI Cliff (median)',
                 f'{cliff["bd_before_ltd"].median():.0f} bd before LTD' if len(cliff) else '—',
                 'OI < 50% of P1 peak', '#f0883e'),
            pill('Today', str(ci["today"]),
                 f'Current position: {cur_pos.get("period","??")} BD{cur_pos.get("bd","?")} ({cur_pos.get("year","?")})',
                 '#8b949e'),
            pill('P1 Duration', f'{b["p1_bd"]} BD',
                 f'{b["p1_start"]} → {b["p1_end"]}'),
            pill('P3 Duration', f'{b["p3_bd"]} BD',
                 f'{b["p3_start"]} → {b["mar_ltd"]}'),
        ], style={'display':'flex','gap':'10px','flexWrap':'wrap','marginBottom':'14px'})

        # Period description block
        descs = html.Div([
            html.Div([
                html.Span('Period 1 (P1)', style={'color':P_COL['P1'],'fontWeight':'500'}),
                html.Span(' — Starts the day after the last BD of November (CC Dec contract LTD). '
                          'Ends the day before the Goldman roll. This is when CC March becomes the '
                          'primary front contract. Typical duration 33–38 BD.', style={'color':MUTED}),
            ], style={'marginBottom':'6px'}),
            html.Div([
                html.Span('Period 2 (P2) — Goldman Roll', style={'color':P_COL['P2'],'fontWeight':'500'}),
                html.Span(' — 5th to 9th BD of February. Index rolls out of March into May (K). '
                          'Rogers OUT roll overlaps (penultimate BD Feb → 1st BD Mar). 5 trading days.', style={'color':MUTED}),
            ], style={'marginBottom':'6px'}),
            html.Div([
                html.Span('Period 3 (P3)', style={'color':P_COL['P3'],'fontWeight':'500'}),
                html.Span(' — Starts day after Goldman roll ends. Ends on CC March LTD. '
                          'Contract enters delivery convergence. OI decays rapidly. Typical 20–25 BD.', style={'color':MUTED}),
            ]),
        ], style={**CARD, 'fontSize':'12px', 'lineHeight':'1.7'})

        # Timeline bar
        timeline = go.Figure()
        for p, col in P_COL.items():
            sk = 'p1_start' if p=='P1' else 'p2_start' if p=='P2' else 'p3_start'
            ek = 'p1_end'   if p=='P1' else 'p2_end'   if p=='P2' else 'mar_ltd'
            bds = count_bd(b[sk], b[ek])
            timeline.add_trace(go.Bar(x=[bds], y=[p], orientation='h',
                marker_color=col, name=p,
                text=f'{p}: {bds} BD | {b[sk]} → {b[ek]}',
                textposition='inside'))
        # Rogers markers
        timeline.add_trace(go.Scatter(
            x=[count_bd(b['p1_start'], b['rogers_out_start'])],
            y=['P2'], mode='markers+text',
            marker=dict(color=RG_COL, size=12, symbol='diamond'),
            text=['Rogers OUT'], textposition='top center',
            name='Rogers OUT'))
        timeline.update_layout(**base_layout(
            title=f'Period Timeline — CC H {year}',
            barmode='stack', height=160,
            margin=dict(l=60,r=20,t=36,b=10),
            showlegend=False,
            xaxis=dict(title='Business Days', gridcolor=BORDER),
            yaxis=dict(gridcolor=BORDER)))

        return [
            pills, descs,
            html.Div([dcc.Graph(figure=timeline)], style=CARD),
            html.Div([
                html.Div('Period Boundaries — All Selected Years',
                         style={'fontSize':'13px','fontWeight':'500','marginBottom':'10px'}),
                make_table(bdf, height='260px')
            ], style=CARD),
            html.Div([dcc.Graph(figure=chart_oi_decay(years_sel))], style=CARD),
        ]

    # ── OUTRIGHT ──────────────────────────────────────────────────────────────
    elif tab == 'tab-out':
        single_fig  = chart_ohlc_oi_vol(year)
        atr_fig     = chart_atr(years_sel, period)
        oi_fig      = chart_oi_by_bd(years_sel, 'out', period)
        vol_fig     = chart_vol_by_bd(years_sel, 'out', period)
        hr_fig      = chart_hit_rate_bar(years_sel, 'out', period)
        nm_fig      = chart_net_move_bar(years_sel, 'out', period)
        oi_dec_fig  = chart_oi_decay(years_sel)

        # Range by period avg
        rr = range_ratio(DATA, years_sel, 'out')
        rr_fig = go.Figure()
        if len(rr):
            for p, col in P_COL.items():
                s = rr[rr['period']==p]
                rr_fig.add_trace(go.Bar(x=s['year'], y=s['avg_range'],
                    name=p, marker_color=col))
        apply_base(rr_fig, title='Avg Daily Range (H-L) by Period & Year',
                   barmode='group', height=280)

        return [
            html.Div([
                html.Div('Single Year — OHLC, OI & Volume with Period Bands & Roll Markers',
                         style={'fontSize':'13px','fontWeight':'500','marginBottom':'8px'}),
                dcc.Graph(figure=single_fig)
            ], style=CARD),

            html.Div([
                html.Div(f'Selected Period: {period}', style={'fontSize':'13px','fontWeight':'500','marginBottom':'8px',
                          'color':P_COL[period]}),
                html.Div([
                    html.Div([dcc.Graph(figure=atr_fig)],  style={'flex':'1'}),
                    html.Div([dcc.Graph(figure=oi_fig)],   style={'flex':'1'}),
                ], style={'display':'flex','gap':'12px'}),
                html.Div([dcc.Graph(figure=vol_fig)]),
            ], style=CARD),

            html.Div([
                html.Div([dcc.Graph(figure=hr_fig)],  style={'flex':'1'}),
                html.Div([dcc.Graph(figure=nm_fig)],  style={'flex':'1'}),
            ], style={**CARD,'display':'flex','gap':'12px'}),

            html.Div([
                html.Div([dcc.Graph(figure=rr_fig)],     style={'flex':'1'}),
                html.Div([dcc.Graph(figure=oi_dec_fig)], style={'flex':'1.2'}),
            ], style={**CARD,'display':'flex','gap':'12px'}),
        ]

    # ── H-K SPREAD ────────────────────────────────────────────────────────────
    elif tab == 'tab-hk':
        single_fig = chart_spread_detail(year, 'hk')
        bd_fig     = chart_spread_by_bd(years_sel, 'hk', period)
        oi_fig     = chart_oi_by_bd(years_sel, 'hk', period)
        vol_fig    = chart_vol_by_bd(years_sel, 'hk', period)
        hr_fig     = chart_hit_rate_bar(years_sel, 'hk', period)
        nm_fig     = chart_net_move_bar(years_sel, 'hk', period)

        return [
            html.Div([
                html.Div('H-K Spread — Single Year Detail (H-L Range Bars + Last Markers)',
                         style={'fontSize':'13px','fontWeight':'500','marginBottom':'8px'}),
                dcc.Graph(figure=single_fig),
            ], style=CARD),
            html.Div([
                html.Div(f'Period {period} — All Selected Years',
                         style={'fontSize':'13px','fontWeight':'500','marginBottom':'8px',
                                'color':P_COL[period]}),
                html.Div([
                    html.Div([dcc.Graph(figure=bd_fig)],  style={'flex':'1.2'}),
                    html.Div([dcc.Graph(figure=hr_fig)],  style={'flex':'1'}),
                ], style={'display':'flex','gap':'12px'}),
                html.Div([
                    html.Div([dcc.Graph(figure=oi_fig)],  style={'flex':'1'}),
                    html.Div([dcc.Graph(figure=vol_fig)], style={'flex':'1'}),
                ], style={'display':'flex','gap':'12px'}),
                dcc.Graph(figure=nm_fig),
            ], style=CARD),
        ]

    # ── HKN FLY ───────────────────────────────────────────────────────────────
    elif tab == 'tab-fly':
        single_fig = chart_fly_detail(year)
        bd_fig     = chart_spread_by_bd(years_sel, 'fly', period)
        vol_fig    = chart_vol_by_bd(years_sel, 'fly', period)
        hr_fig     = chart_hit_rate_bar(years_sel, 'fly', period)
        nm_fig     = chart_net_move_bar(years_sel, 'fly', period)

        return [
            html.Div([
                html.Div('HKN Fly (H − 2K + N) — Single Year Detail',
                         style={'fontSize':'13px','fontWeight':'500','marginBottom':'8px'}),
                dcc.Graph(figure=single_fig),
            ], style=CARD),
            html.Div([
                html.Div(f'Period {period} — All Selected Years',
                         style={'fontSize':'13px','fontWeight':'500','marginBottom':'8px',
                                'color':P_COL[period]}),
                html.Div([
                    html.Div([dcc.Graph(figure=bd_fig)], style={'flex':'1.2'}),
                    html.Div([dcc.Graph(figure=hr_fig)], style={'flex':'1'}),
                ], style={'display':'flex','gap':'12px'}),
                html.Div([dcc.Graph(figure=vol_fig)]),
                dcc.Graph(figure=nm_fig),
            ], style=CARD),
        ]

    # ── ROLL WINDOW ───────────────────────────────────────────────────────────
    elif tab == 'tab-roll':
        # P2 spread price: Open of first day, Last of last day
        roll_rows = []
        for yr in sorted(years_sel):
            b = DATA[yr]['bounds']
            for ser, lbl in [('hk','H-K'),('kn','K-N')]:
                sub = DATA[yr].get(ser, pd.DataFrame())
                p2  = sub[(sub['period']=='P2') & sub['close_price'].notna()].sort_values('date')
                if len(p2) < 2: continue
                first = p2.iloc[0]; last = p2.iloc[-1]
                op    = first['Open'] if pd.notna(first.get('Open')) else first['close_price']
                cl    = last['Last']  if pd.notna(last.get('Last'))  else last['close_price']
                net   = cl - op
                roll_rows.append({'Year':yr,'Spread':lbl,
                    'Goldman Start':str(b['p2_start']),'Goldman End':str(b['p2_end']),
                    'Rogers OUT Start':str(b['rogers_out_start']),'Rogers OUT End':str(b['rogers_out_end']),
                    'Open (bd1)':f'{op:.0f}','Last (bd5)':f'{cl:.0f}',
                    'Net Move':f'{net:+.0f}','Direction':'▲' if net>0 else '▼'})
        roll_df = pd.DataFrame(roll_rows)

        # Avg spread move per roll BD
        hk_roll = spread_move_vs_roll(DATA, years_sel, 'hk')
        kn_roll = spread_move_vs_roll(DATA, years_sel, 'kn')

        roll_fig = make_subplots(rows=1, cols=2,
            subplot_titles=['H-K Spread: Avg Δ per Roll BD',
                            'K-N Spread: Avg Δ per Roll BD'])
        for df_r, ci in [(hk_roll, 1),(kn_roll, 2)]:
            if len(df_r) == 0: continue
            clrs = ['#3fb950' if v >= 0 else '#da3633' for v in df_r['avg_chg']]
            roll_fig.add_trace(go.Bar(
                x=df_r['bd'], y=df_r['avg_chg'],
                error_y=dict(type='data', array=df_r['std_chg'], visible=True),
                marker_color=clrs,
                text=[f'{v:+.1f}' for v in df_r['avg_chg']], textposition='outside',
                name='Avg Δ'), row=1, col=ci)
        apply_base(roll_fig, title='Goldman Roll Period — Avg Spread Move per BD', height=320)

        # OI & Volume in P2 by BD
        oi_p2_fig = go.Figure()
        vol_p2_fig = go.Figure()
        rows_oi = []; rows_vol = []
        for yr in years_sel:
            sub = DATA[yr]['out']
            p2 = sub[(sub['period']=='P2')].sort_values('date')
            if 'oi_chg' in p2.columns:
                p2c = p2[p2['oi_chg'].notna()][['bd','oi_chg']]; p2c['year'] = yr
                rows_oi.append(p2c)
            if 'Volume' in p2.columns:
                p2v = p2[p2['Volume'].notna()][['bd','Volume']]; p2v['year'] = yr
                rows_vol.append(p2v)
        if rows_oi:
            combined = pd.concat(rows_oi)
            agg = combined.groupby('bd')['oi_chg'].agg(['mean','std']).reset_index()
            clrs = ['#3fb950' if v >= 0 else '#da3633' for v in agg['mean']]
            oi_p2_fig.add_trace(go.Bar(x=agg['bd'], y=agg['mean'],
                error_y=dict(type='data', array=agg['std'].fillna(0), visible=True),
                marker_color=clrs, name='Avg OI Δ'))
        apply_base(oi_p2_fig, title='P2 Outright OI Change per Roll BD (avg)', height=260)
        if rows_vol:
            combined = pd.concat(rows_vol)
            agg = combined.groupby('bd')['Volume'].mean().reset_index()
            vol_p2_fig.add_trace(go.Bar(x=agg['bd'], y=agg['Volume'],
                marker_color='rgba(240,136,62,0.7)', name='Avg Volume'))
        apply_base(vol_p2_fig, title='P2 Volume per Roll BD (avg)', height=260)

        # Fly in P2
        fly_p2_fig = chart_spread_by_bd(years_sel, 'fly', 'P2')

        return [
            html.Div([
                html.Div([
                    html.Span('Goldman Roll', style={'color':GS_COL,'fontWeight':'500'}),
                    html.Span(' — 5th to 9th BD of February.  ', style={'color':MUTED}),
                    html.Span('Rogers OUT Roll', style={'color':RG_COL,'fontWeight':'500'}),
                    html.Span(' — Penultimate BD of Feb → 1st BD of March. These overlap at the tail of P2.',
                              style={'color':MUTED}),
                ], style={'fontSize':'12px','marginBottom':'12px'}),
                make_table(roll_df, conditional=UP_STYLE+DOWN_STYLE, height='280px'),
            ], style=CARD),
            html.Div([dcc.Graph(figure=roll_fig)], style=CARD),
            html.Div([
                html.Div([dcc.Graph(figure=oi_p2_fig)],  style={'flex':'1'}),
                html.Div([dcc.Graph(figure=vol_p2_fig)], style={'flex':'1'}),
            ], style={**CARD,'display':'flex','gap':'12px'}),
            html.Div([
                html.Div('HKN Fly during Goldman Roll Window (all years)',
                         style={'fontSize':'13px','fontWeight':'500','marginBottom':'8px'}),
                dcc.Graph(figure=fly_p2_fig)
            ], style=CARD),
        ]

    # ── WEEK ANALYSIS ─────────────────────────────────────────────────────────
    elif tab == 'tab-week':
        ws_df = week_stats(DATA, years_sel, series, period)
        if len(ws_df) == 0:
            return [html.Div('No data for selection.', style={'color':MUTED})]

        # Week move by year heatmap
        pivot_move = ws_df.pivot_table(index='week_in_period', columns='year', values='week_move', aggfunc='mean')
        heat_fig = go.Figure(go.Heatmap(
            z=pivot_move.values,
            x=[str(c) for c in pivot_move.columns],
            y=[f'Wk{r}' for r in pivot_move.index],
            colorscale='RdYlGn', zmid=0,
            text=[[f'{v:+.0f}' if not np.isnan(v) else '' for v in row] for row in pivot_move.values],
            texttemplate='%{text}', hovertemplate='%{y} %{x}: %{z:+.0f}<extra></extra>',
        ))
        apply_base(heat_fig, title=f'{series.upper()} {period} — Weekly Net Move Heatmap (pts)',
                   height=320)
        heat_fig.update_layout(xaxis=dict(title='Year', rangebreaks=[]))
        heat_fig.update_xaxes(rangebreaks=[])

        # Week avg move bar
        wk_avg = ws_df.groupby('week_in_period')['week_move'].agg(['mean','std']).reset_index()
        clrs = ['#3fb950' if v >= 0 else '#da3633' for v in wk_avg['mean']]
        wk_bar = go.Figure()
        wk_bar.add_trace(go.Bar(x=[f'Wk{r}' for r in wk_avg['week_in_period']],
            y=wk_avg['mean'],
            error_y=dict(type='data', array=wk_avg['std'].fillna(0), visible=True),
            marker_color=clrs,
            text=[f'{v:+.0f}' for v in wk_avg['mean']], textposition='outside'))
        apply_base(wk_bar, title=f'{series.upper()} {period} — Avg Weekly Move across years',
                   yaxis_title='Points', height=280)
        wk_bar.update_layout(xaxis=dict(rangebreaks=[]))
        wk_bar.update_xaxes(rangebreaks=[])

        # Volume by week
        wk_vol = ws_df.groupby(['week_in_period','year'])['avg_vol'].mean().reset_index()
        vol_heat_pivot = wk_vol.pivot_table(index='week_in_period', columns='year', values='avg_vol')
        vol_heat = go.Figure(go.Heatmap(
            z=vol_heat_pivot.values,
            x=[str(c) for c in vol_heat_pivot.columns],
            y=[f'Wk{r}' for r in vol_heat_pivot.index],
            colorscale='Blues',
            hovertemplate='%{y} %{x}: %{z:,.0f}<extra></extra>',
        ))
        apply_base(vol_heat, title=f'{series.upper()} {period} — Weekly Avg Volume Heatmap',
                   height=300)
        vol_heat.update_layout(xaxis=dict(rangebreaks=[]))
        vol_heat.update_xaxes(rangebreaks=[])

        # WoW comparison table (pivot: week × year)
        tbl_df = ws_df[['year','week_in_period','week_open','week_close','week_move','week_pct','avg_vol','n_days']].copy()
        tbl_df.columns = ['Year','Week','Open','Close','Net Move','% Move','Avg Vol','Days']
        tbl_df = tbl_df.round(1).astype(str)

        return [
            html.Div([
                html.Div([dcc.Graph(figure=heat_fig)], style={'flex':'1.2'}),
                html.Div([dcc.Graph(figure=wk_bar)],  style={'flex':'1'}),
            ], style={**CARD,'display':'flex','gap':'12px'}),
            html.Div([dcc.Graph(figure=vol_heat)], style=CARD),
            html.Div([
                html.Div('Week-by-Week Detail Table',
                         style={'fontSize':'13px','fontWeight':'500','marginBottom':'10px'}),
                make_table(tbl_df, height='320px'),
            ], style=CARD),
        ]

    # ── BD CALENDAR ───────────────────────────────────────────────────────────
    elif tab == 'tab-cal':
        bmap_filt = BD_MAP[BD_MAP['Year'].isin(years_sel)].copy()

        # Pivot: BD vs Year for selected period
        piv = bmap_filt[bmap_filt['Period']==period].pivot_table(
            index='BD', columns='Year', values='Date', aggfunc='first')
        piv = piv.reset_index()
        piv.columns = [str(c) for c in piv.columns]

        ci = current_bd_info(DATA, years_sel)
        today_str = str(ci['today'])

        # Highlight today's equivalent
        style_cond = []
        for pos in ci['positions']:
            if pos['period'] == period:
                style_cond.append({
                    'if': {'filter_query': f'{{BD}} = {pos["bd"]}',
                           'column_id': str(pos['year'])},
                    'backgroundColor': '#1f6feb', 'color': 'white', 'fontWeight': 'bold'
                })

        return [
            html.Div([
                html.Div([
                    html.Span('BD Calendar for Period ', style={'color':MUTED}),
                    html.Span(period, style={'color':P_COL[period],'fontWeight':'500'}),
                    html.Span(' — each cell shows the calendar date for that BD in that year.',
                              style={'color':MUTED}),
                    html.Span(f' Today = {today_str}. Blue cell = equivalent position today.',
                              style={'color':'#388bfd'}),
                ], style={'fontSize':'12px','marginBottom':'10px'}),
                make_table(piv, conditional=style_cond, height='500px'),
            ], style=CARD),
            html.Div([
                html.Div('Full BD Map (all periods, all years)',
                         style={'fontSize':'13px','fontWeight':'500','marginBottom':'10px'}),
                make_table(bmap_filt[['Year','Period','BD','Week','Date','Day','Calendar Week']],
                           height='400px'),
            ], style=CARD),
        ]

    # ── PATTERN ENGINE ────────────────────────────────────────────────────────
    elif tab == 'tab-pat':
        return build_pattern_tab(years_sel, period, series)

    return html.Div('Select a tab.')



def render_finding_card(f):
    """Render a single pattern finding as a styled Dash card."""
    conf_border = {'HIGH': '#2ea043', 'MEDIUM': '#d29922', 'LOW': '#6e7681'}
    conf_bg     = {'HIGH': '#0d1a0d', 'MEDIUM': '#1a1500', 'LOW': '#161b22'}
    border_col  = conf_border[f['confidence']]
    bg_col      = conf_bg[f['confidence']]

    badge_style = {
        'background': border_col, 'color': '#fff',
        'borderRadius': '3px', 'padding': '1px 6px',
        'fontSize': '10px', 'fontWeight': '700',
        'marginRight': '8px', 'letterSpacing': '.05em',
        'textTransform': 'uppercase', 'flexShrink': '0',
    }
    ptype_badge = {
        'background': f.get('ptype_color', '#388bfd') + '22',
        'color': f.get('ptype_color', '#388bfd'),
        'borderRadius': '3px', 'padding': '1px 7px',
        'fontSize': '10px', 'fontWeight': '500',
        'marginRight': '6px', 'flexShrink': '0',
    }
    hit_str = f'{f["hits"]}/{f["n_years"]} ({f["hit_rate"]:.0%})'

    extras = []
    if f.get('volume_context'):
        extras.append(html.Div([
            html.Span('VOL ', style={'color':'#f0883e','fontWeight':'600','fontSize':'10px','marginRight':'4px'}),
            html.Span(f['volume_context'], style={'color':'#c9d1d9','fontSize':'11px'}),
        ], style={'marginTop':'4px'}))
    if f.get('oi_context'):
        extras.append(html.Div([
            html.Span('OI  ', style={'color':'#388bfd','fontWeight':'600','fontSize':'10px','marginRight':'4px'}),
            html.Span(f['oi_context'], style={'color':'#c9d1d9','fontSize':'11px'}),
        ], style={'marginTop':'3px'}))
    if f.get('additional_note'):
        extras.append(html.Div([
            html.Span('💡 ', style={'fontSize':'11px'}),
            html.Span(f['additional_note'], style={'color':'#bc8cff','fontSize':'11px','fontStyle':'italic'}),
        ], style={'marginTop':'3px'}))

    avg_m_str = f'  avg {f["avg_move"]:+.0f} pts' if f.get('avg_move') is not None else ''

    return html.Div([
        # Header
        html.Div([
            html.Span(f['confidence'], style=badge_style),
            html.Span(f['pattern_type'], style=ptype_badge),
            html.Span(
                f'{f["series_label"]} · {f["period_label"]}',
                style={'color': MUTED, 'fontSize': '11px', 'flex': '1'}
            ),
            html.Span(
                hit_str + avg_m_str,
                style={'color': border_col, 'fontWeight': '700', 'fontSize': '12px', 'flexShrink': '0'}
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '4px', 'marginBottom': '7px'}),

        # Description
        html.P(f['description'],
               style={'color': '#c9d1d9', 'fontSize': '12px', 'margin': '0 0 6px', 'lineHeight': '1.6'}),

        # Extra context
        *extras,

        # Years bar
        html.Div([
            html.Span('✓ ', style={'color': border_col, 'fontWeight': '600', 'fontSize': '11px'}),
            *[html.Span(str(yr) + ' ', style={
                'color': border_col, 'fontSize': '11px',
                'background': border_col + '22',
                'borderRadius': '3px', 'padding': '0 4px', 'marginRight': '3px',
              }) for yr in f['years_hit']],
        ], style={'marginTop': '6px'}),
        html.Div([
            html.Span('✗ ', style={'color': '#6e7681', 'fontSize': '11px'}),
            html.Span(' '.join(str(y) for y in f['years_miss']) or '—',
                      style={'color': '#6e7681', 'fontSize': '11px'}),
        ], style={'marginTop': '3px'}),
    ], style={
        'background': bg_col,
        'border': f'1px solid {border_col}44',
        'borderLeft': f'4px solid {border_col}',
        'borderRadius': '6px',
        'padding': '12px 14px',
        'marginBottom': '8px',
    })


def build_pattern_tab(years_sel, period, series):
    from pattern_engine_v2 import (
        run_all_patterns, find_similar_years, SERIES_LABELS, PERIOD_LABELS
    )

    # ── Run all pattern detectors ─────────────────────────────────────────────
    all_findings = CACHED_PATTERNS
    n_high   = sum(1 for f in all_findings if f['confidence']=='HIGH')
    n_medium = sum(1 for f in all_findings if f['confidence']=='MEDIUM')
    n_low    = sum(1 for f in all_findings if f['confidence']=='LOW')

    # Similar years
    sim_years = find_similar_years(DATA, YEARS, ref_year=max(YEARS), n_similar=3)
    sim_txt = ', '.join(f'{yr} (score {sc:.2f})' for yr, sc in sim_years.items()) if sim_years else '—'

    # ── Summary bar ───────────────────────────────────────────────────────────
    summary_bar = html.Div([
        html.Div([
            html.Div(str(n_high),   style={'color':'#2ea043','fontSize':'24px','fontWeight':'700'}),
            html.Div('HIGH',        style={'color':MUTED,'fontSize':'10px','textTransform':'uppercase'}),
        ], style={**CARD,'padding':'10px 16px','textAlign':'center','flex':'0 0 80px'}),
        html.Div([
            html.Div(str(n_medium), style={'color':'#d29922','fontSize':'24px','fontWeight':'700'}),
            html.Div('MEDIUM',      style={'color':MUTED,'fontSize':'10px','textTransform':'uppercase'}),
        ], style={**CARD,'padding':'10px 16px','textAlign':'center','flex':'0 0 80px'}),
        html.Div([
            html.Div(str(n_low),    style={'color':'#6e7681','fontSize':'24px','fontWeight':'700'}),
            html.Div('LOW',         style={'color':MUTED,'fontSize':'10px','textTransform':'uppercase'}),
        ], style={**CARD,'padding':'10px 16px','textAlign':'center','flex':'0 0 80px'}),
        html.Div([
            html.Div(f'{len(all_findings)} total', style={'color':TEXT,'fontSize':'16px','fontWeight':'500'}),
            html.Div('patterns detected across all instruments × periods',
                     style={'color':MUTED,'fontSize':'11px'}),
            html.Div(f'Closest analog to {max(years_sel) if years_sel else "?"}: {sim_txt}',
                     style={'color':'#bc8cff','fontSize':'11px','marginTop':'3px'}),
        ], style={**CARD,'flex':'1','padding':'10px 16px'}),
    ], style={'display':'flex','gap':'10px','marginBottom':'4px'})

    # ── Filter controls ───────────────────────────────────────────────────────
    all_ptypes = sorted(set(f['pattern_type'] for f in all_findings))
    filters = html.Div([
        html.Div([
            ctrl_label('Confidence'),
            dcc.Checklist(
                id='pat-conf-filter',
                options=[{'label': c, 'value': c} for c in ['HIGH','MEDIUM','LOW']],
                value=['HIGH','MEDIUM'],
                inline=True,
                labelStyle={'marginRight':'12px','fontSize':'12px','color':TEXT},
            ),
        ], style={'marginRight':'20px'}),
        html.Div([
            ctrl_label('Instrument'),
            dcc.Checklist(
                id='pat-series-filter',
                options=[{'label': v, 'value': k} for k,v in SERIES_LABELS.items() if k in ['out','hk','fly']],
                value=['out','hk','fly'],
                inline=True,
                labelStyle={'marginRight':'10px','fontSize':'12px','color':TEXT},
            ),
        ], style={'marginRight':'20px'}),
        html.Div([
            ctrl_label('Period'),
            dcc.Checklist(
                id='pat-period-filter',
                options=[{'label': p, 'value': p} for p in ['P1','P2','P3','P1→P2','P2→P3']],
                value=['P1','P2','P3','P1→P2','P2→P3'],
                inline=True,
                labelStyle={'marginRight':'10px','fontSize':'12px','color':TEXT},
            ),
        ]),
    ], style={'display':'flex','flexWrap':'wrap','gap':'8px',
              'padding':'12px 14px','background':CARD_BG,'borderRadius':'6px',
              'border':f'1px solid {BORDER}','marginBottom':'14px'})

    # ── Apply filters and render cards ────────────────────────────────────────
    # Default: show HIGH + MEDIUM, all series, all periods
    filtered = [f for f in all_findings
                if f['confidence'] in ['HIGH','MEDIUM']
                and f['series'] in ['out','hk','fly']
                and f['period'] in ['P1','P2','P3','P1→P2','P2→P3']]

    # Group by period for section headers
    period_order = ['P1','P1→P2','P2','P2→P3','P3']
    by_period = {}
    for f in filtered:
        by_period.setdefault(f['period'], []).append(f)

    sections = []
    for p in period_order:
        pf = by_period.get(p, [])
        if not pf: continue
        p_label = PERIOD_LABELS.get(p, p)
        p_color = {'P1':P_COL['P1'],'P2':P_COL['P2'],'P3':P_COL['P3'],
                   'P1→P2':'#888','P2→P3':'#888'}.get(p, '#888')
        sections.append(html.Div([
            html.Div(f'{p_label} — {len(pf)} findings',
                     style={'color': p_color, 'fontSize': '13px', 'fontWeight': '600',
                            'marginBottom': '8px', 'paddingBottom': '6px',
                            'borderBottom': f'1px solid {p_color}44'}),
            *[render_finding_card(f) for f in pf],
        ], style={'marginBottom': '18px'}))

    cards_section = html.Div([
        html.Div(
            f'Showing {len(filtered)} findings (HIGH + MEDIUM confidence) — '
            f'adjust filters above to show LOW confidence patterns',
            style={'color': MUTED, 'fontSize': '11px', 'marginBottom': '12px'}
        ),
        *sections,
    ])

    # ── Existing reference charts (condensed) ─────────────────────────────────
    bias_data = []
    for ser, lbl in [('out','Outright'),('hk','H-K Spread'),('fly','HKN Fly')]:
        for p in ['P1','P2','P3']:
            _, st = directional_bias(DATA, years_sel, ser, 'close_price', p)
            if st:
                bias_data.append({'Series':lbl,'Period':p,'Hit Rate':st['hit_rate'],
                                   'Direction':st['direction'],'Avg Move':st['avg_move']})
    bias_heat = go.Figure()
    if bias_data:
        bdf = pd.DataFrame(bias_data)
        pivot_b = bdf.pivot(index='Series',columns='Period',values='Hit Rate')
        bias_heat = go.Figure(go.Heatmap(
            z=pivot_b.values, x=list(pivot_b.columns), y=list(pivot_b.index),
            colorscale='RdYlGn', zmin=0.4, zmax=1.0,
            text=[[f'{v:.0%}' for v in row] for row in pivot_b.values],
            texttemplate='%{text}',
        ))
        apply_base(bias_heat, title='Directional Bias Hit Rate', height=220)
        bias_heat.update_layout(xaxis=dict(rangebreaks=[]))
        bias_heat.update_xaxes(rangebreaks=[])

    sc = seasonal_scorecard(DATA, years_sel)
    sc_cols = ['year','out_P1','out_P2','out_P3','hk_P1','hk_P2','hk_P3','fly_P1','fly_P2','fly_P3']
    sc_cols = [c for c in sc_cols if c in sc.columns]
    sc_cond = []
    for c in sc_cols:
        if c == 'year': continue
        sc_cond += [
            {'if':{'filter_query':f'{{{c}}} = "▲"','column_id':c},'color':'#3fb950','fontWeight':'bold'},
            {'if':{'filter_query':f'{{{c}}} = "▼"','column_id':c},'color':'#da3633','fontWeight':'bold'},
        ]

    return [
        summary_bar,
        filters,
        html.Div([
            html.Div('Auto-Detected Patterns — Sorted by Confidence then Hit Rate',
                     style={'fontSize':'14px','fontWeight':'500','marginBottom':'12px'}),
            cards_section,
        ], style=CARD),
        html.Div([
            html.Div([dcc.Graph(figure=bias_heat)], style={'flex':'1'}),
            html.Div([
                html.Div('Seasonal Scorecard',
                         style={'fontSize':'13px','fontWeight':'500','marginBottom':'8px'}),
                make_table(sc[sc_cols], conditional=sc_cond, height='220px'),
            ], style={'flex':'2'}),
        ], style={**CARD,'display':'flex','gap':'12px'}),
    ]

server = app.server
if __name__ == '__main__':
    print('\n' + '='*55)
    print('  CC March Dashboard v2')
    print('  http://127.0.0.1:8050')
    print('='*55 + '\n')
    app.run(debug=False, host='127.0.0.1', port=8050)
