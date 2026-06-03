# -*- coding: utf-8 -*-
# 🌀 Surface Synoptic Map — Western Canada
# Fetches live METARs · Kriging/RBF interpolation · 4 hPa SLP isobars · WMO station models
# Source: https://raw.githubusercontent.com/ngsmetadvisor/SfcMap/main/AP_location.csv
# METAR source: https://aviationweather.gov/api/data/metar

# ── Cell 1 . Install & import packages ────────────────────────
import subprocess, sys, importlib, importlib.util

# ── Headless shim: replace Colab display utilities ────────────────────────────
import sys as _sys
class _FakeDisplay:
    def __init__(self, *a, **kw): pass
    def _repr_html_(self): return ''
def display(*a, **kw): pass
class HTML(_FakeDisplay): pass
class SVG(_FakeDisplay): pass
# ──────────────────────────────────────────────────────────────────────────────

# ── Install missing packages ──────────────────────────────────
_pkg_map = {
    'requests':   'requests',
    'scipy':      'scipy',
    'matplotlib': 'matplotlib',
    'numpy':      'numpy',
    'pykrige':    'pykrige',
    'folium':     'folium',
    'shapely':    'shapely',
    'branca':     'branca',
}

_to_install = [pip for pip, imp in _pkg_map.items()
               if importlib.util.find_spec(imp) is None]

if _to_install:
    print(f'Installing: {_to_install}')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + _to_install)
else:
    print('All packages already present — nothing to install.')

# ── Standard library ──────────────────────────────────────────
import csv, io, json, math, re, time, warnings
import os as _os; _os.makedirs("output", exist_ok=True)
import concurrent.futures
from collections import defaultdict

# ── Third-party: core ─────────────────────────────────────────
import numpy as np
import pandas as pd
import requests

# ── Third-party: matplotlib (backend before pyplot) ───────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Third-party: scipy ────────────────────────────────────────
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import (gaussian_filter, maximum_filter,
                           minimum_filter, label)

# ── Third-party: geo / mapping ────────────────────────────────
import folium
import branca
from shapely.geometry import shape

# ── Third-party: kriging ──────────────────────────────────────
from pykrige.ok import OrdinaryKriging

print('✓ All packages ready')

# ── Cell 1.5 . Configuration ──────────────────────────────────

# ── Data sources ──────────────────────────────────────────────
CSV_URL   = 'https://raw.githubusercontent.com/ngsmetadvisor/SfcMap/main/AP_location.csv'
METAR_API = 'https://aviationweather.gov/api/data/metar'
COVERAGE  = 'chart'          # essential | standard | all | chart
EXPORT_TIME = '1200Z'        # 0000Z | 0600Z | 1200Z | 1800Z

# ── Interpolation ─────────────────────────────────────────────
INTERP_METHOD = 'rbf'    # rbf | kriging
GRID_N        = 300          # grid points per axis  (60–600, step 20)
RBF_SMOOTHING = 0.005         # 0.0 = exact fit; 0.2–0.5 typical for SLP
SIGMA_SMOOTH  = 0.2          # gaussian blur after interpolation (2–4 typical)

# ── Isobars ───────────────────────────────────────────────────
SLP_INTERVAL  = 4            # hPa between isobars (standard = 4)

# ── H / L centre detection ────────────────────────────────────
# HL_SIGMA        extra Gaussian blur before peak search (stacks on SIGMA_SMOOTH)
#                 low=sensitive/noisy  high=smooth/may miss weak systems  typical: 2–4
# HL_NEIGHBORHOOD min grid-cell separation between two H or L centres
#                 20cells≈250km  30cells≈375km  typical synoptic: 25–35
# HL_MIN_DELTA    min hPa relief (peak vs neighbourhood floor/ceiling)
#                 0.5=very sensitive  3–4=standard  6+=strong systems only
HL_SIGMA        = 2.0
HL_NEIGHBORHOOD = 20
HL_MIN_DELTA    = 2.0

# ── Station model rendering ───────────────────────────────────
SYMBOL_SCALE  = 28           # station symbol size px (10–80)
FONT_SCALE    = 10           # station label font size (4–20)

print(f'Coverage: {COVERAGE} | Interp: {INTERP_METHOD} | '
      f'Grid: {GRID_N} | Export: {EXPORT_TIME}')


import os
from datetime import datetime, timezone

_FLAG_FILE = 'output/_auto_export_flag'

def set_auto_export():
    """Call this once to arm the auto-export, then Run All."""
    with open(_FLAG_FILE, 'w') as f:
        f.write(datetime.now(timezone.utc).isoformat())
    print('Auto-export ARMED — now Run All.')

# Check if flag exists and is less than 2 hours old
_AUTO_EXPORT = False
if os.path.exists(_FLAG_FILE):
    with open(_FLAG_FILE) as f:
        _flag_time = datetime.fromisoformat(f.read().strip())
    _age_minutes = (datetime.now(timezone.utc) - _flag_time).total_seconds() / 60
    if _age_minutes <= 120:
        _AUTO_EXPORT = True
        print(f'Auto-export ACTIVE (flag is {_age_minutes:.0f} min old)')
        os.remove(_FLAG_FILE)  # consume the flag — one shot only
    else:
        _AUTO_EXPORT = False
        os.remove(_FLAG_FILE)  # expired, clean up
        print(f'Auto-export flag EXPIRED ({_age_minutes:.0f} min old) — skipping')
else:
    print('Auto-export OFF')

_AUTO_EXPORT = False  # auto-download disabled; use the Download HTML button instead

# -- Cell 8 - WMO station model as SVG string ---
import math

_CR = 0.14   # smaller station circle


# ── FEATHER ANGLE CONTROL ──────────────────────────────────────────────────
# Angle of feathers relative to staff (degrees).
# 90  = perpendicular to staff (standard WMO)
# >90 = feathers tilt AWAY from circle (toward tip)
# <90 = feathers tilt TOWARD circle
FEATHER_ANGLE = 110   # ← change this value to adjust feather angle

# Side of feather: +1 = right side of staff (looking from base to tip)
#                  -1 = left side  (standard WMO)
FEATHER_SIDE = +1    # ← change to +1 to flip to right side
# ──────────────────────────────────────────────────────────────────────────


def cloud_circle_svg(cx, cy, R, oktas):
    lw = max(0.9, R * 0.13)
    s = []
    if oktas == 9:  # VV: full black + white X
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="black" stroke="black" stroke-width="{lw}"/>')
        s.append(f'<line x1="{cx-R*.55:.2f}" y1="{cy-R*.55:.2f}" x2="{cx+R*.55:.2f}" y2="{cy+R*.55:.2f}" stroke="white" stroke-width="{lw*.85:.2f}"/>')
        s.append(f'<line x1="{cx+R*.55:.2f}" y1="{cy-R*.55:.2f}" x2="{cx-R*.55:.2f}" y2="{cy+R*.55:.2f}" stroke="white" stroke-width="{lw*.85:.2f}"/>')
        return ''.join(s)
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="white" stroke="black" stroke-width="{lw}"/>')
    if oktas <= 0:
        return ''.join(s)
    if oktas >= 8:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="black" stroke="black" stroke-width="{lw}"/>')
        return ''.join(s)
    if oktas == 2:
        s.append(f'<path d="M{cx},{cy} L{cx},{cy-R:.2f} A{R:.2f},{R:.2f} 0 0,1 {cx+R:.2f},{cy} Z" fill="black"/>')
    elif oktas == 4:
        s.append(f'<path d="M{cx},{cy} L{cx},{cy-R:.2f} A{R:.2f},{R:.2f} 0 1,1 {cx},{cy+R:.2f} Z" fill="black"/>')
    elif oktas == 6:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="black" stroke="black" stroke-width="{lw}"/>')
        s.append(f'<path d="M{cx},{cy} L{cx-R:.2f},{cy} A{R:.2f},{R:.2f} 0 0,1 {cx},{cy-R:.2f} Z" fill="white"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="black" stroke-width="{lw}"/>')
    return ''.join(s)


def wind_barb_svg(cx, cy, R, wind_dir, wind_spd, wind_gust, S):
    """
    WMO wind barb using SVG transform rotate.
    Draws canonical FROM-NORTH barb, then rotates by wind_dir degrees.

    Feather direction and angle controlled by module-level constants:
      FEATHER_SIDE  : -1 = left (WMO standard), +1 = right
      FEATHER_ANGLE : degrees from staff (90=perpendicular, >90=tilts away from circle)
    """
    if wind_dir is None or wind_spd is None:
        return ''
    if wind_spd < 3:
        return (f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{R*1.5:.2f}" '
                f'fill="none" stroke="black" stroke-width="1"/>')

    sl        = S * 1.0    # staff length
    blen      = S * 0.30    # full barb (10-kt feather) length
    blen_penn = S * 0.45    # pennant (50-kt triangle) width
    bspc      = S * 0.115   # spacing between barbs along staff
    lw        = max(0.9, S * 0.038)  # line width

    staff_base_y = -R
    staff_tip_y  = -(R + sl)

    # Feather x-endpoint and y-tilt from FEATHER_SIDE and FEATHER_ANGLE
    # x goes FEATHER_SIDE direction; y tilt = tan(angle-90) * blen toward tip (-y)
    fx_full = FEATHER_SIDE * blen
    fx_half = FEATHER_SIDE * blen * 0.5
    # tilt: >90 deg tilts feather end toward tip (negative y = up)
    tilt = math.tan(math.radians(FEATHER_ANGLE - 90)) * blen

    spd = int(round(wind_spd / 5.0)) * 5
    pn  = spd // 50;  spd -= pn * 50
    fu  = spd // 10;  spd -= fu * 10
    ha  = spd //  5

    parts = []
    parts.append(f'<line x1="0" y1="{staff_base_y:.2f}" x2="0" y2="{staff_tip_y:.2f}" '
                 f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>')

    pos = 0.0

    if pn == 0 and fu == 0 and ha == 1:
        hy = staff_tip_y + 0.28 * sl
        parts.append(f'<line x1="0" y1="{hy:.2f}" x2="{fx_half:.2f}" y2="{hy - tilt*0.5:.2f}" '
                     f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>')
    else:
        for _ in range(pn):  # 50-kt pennants
            ay  = staff_tip_y + pos
            by2 = staff_tip_y + pos + bspc * 2
            pts = f'0,{ay:.2f} {fx_full:.2f},{ay - tilt:.2f} 0,{by2:.2f}'
            parts.append(f'<polygon points="{pts}" fill="black"/>')
            pos += bspc * 1.5
        for _ in range(fu):  # 10-kt full barbs
            fy = staff_tip_y + pos
            parts.append(f'<line x1="0" y1="{fy:.2f}" x2="{fx_full:.2f}" y2="{fy - tilt:.2f}" '
                         f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>')
            pos += bspc
        for _ in range(ha):  # 5-kt half barbs
            hy = staff_tip_y + pos
            parts.append(f'<line x1="0" y1="{hy:.2f}" x2="{fx_half:.2f}" y2="{hy - tilt*0.5:.2f}" '
                         f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>')
            pos += bspc

    inner = ''.join(parts)
    return (f'<g transform="translate({cx:.2f},{cy:.2f}) rotate({wind_dir:.1f})">'
            f'{inner}</g>')

def pressure_tendency_svg(cx, cy, R, tendency, S):
    """
    WMO pressure tendency characteristic symbol, drawn to the right of the
    station circle, vertically centred on the SLP label row.

    tendency codes (WMO):
      0 = rising then falling  (∧  inverted-V)
      1 = rising then steady   (⌐)
      2 = rising               (/)
      3 = falling then rising  - not listed but keep slot
      4 = steady               (—)
      5 = falling then rising  (V)
      6 = falling then steady  (∟)
      7 = falling              (\\)
      8 = steady then falling  (not common, map to 7)

    Also accepts string keys: 'rising', 'falling', 'steady',
      'rising_falling', 'falling_rising', 'rising_steady', 'falling_steady'
    """
    _map = {
        'rising':          2,
        'falling':         7,
        'steady':          4,
        'rising_falling':  0,
        'falling_rising':  5,
        'rising_steady':   1,
        'falling_steady':  6,
    }
    if isinstance(tendency, str):
        tendency = _map.get(tendency.lower(), None)
    if tendency is None:
        return ''

    lw  = max(0.9, S * 0.042)
    # position: right of circle, on the SLP-label row (slightly below centre)
    ox  = cx + R + S * 0.09 + S * 0.52   # shifted right to leave room for change amount
    slp_y = cy - R * 0.6 - 7         # matches slp_label y
    oy  = slp_y + S * 0.55           # one row below SLP label

    arm = S * 0.22    # half-width of symbol
    rise = S * 0.20   # vertical rise of symbol

    def line(x1, y1, x2, y2):
        return (f'<line x1="{ox+x1:.2f}" y1="{oy+y1:.2f}" '
                f'x2="{ox+x2:.2f}" y2="{oy+y2:.2f}" '
                f'stroke="black" stroke-width="{lw:.2f}" '
                f'stroke-linecap="round" stroke-linejoin="round"/>')

    parts = []

    if tendency == 2:        # Rising  /
        parts.append(line(-arm,  rise*0.5, arm, -rise*0.5))

    elif tendency == 7:      # Falling  \
        parts.append(line(-arm, -rise*0.5, arm,  rise*0.5))

    elif tendency == 4:      # Steady  —
        parts.append(line(-arm, 0, arm, 0))

    elif tendency == 0:      # Rising then falling  ∧
        parts.append(line(-arm,  rise*0.5,   0, -rise*0.5))
        parts.append(line(  0, -rise*0.5,  arm,  rise*0.5))

    elif tendency == 5:      # Falling then rising  V
        parts.append(line(-arm, -rise*0.5,   0,  rise*0.5))
        parts.append(line(  0,  rise*0.5,  arm, -rise*0.5))

    elif tendency == 1:      # Rising then steady  ⌐
        parts.append(line(-arm,  rise*0.5,   0, -rise*0.5))   # rising stroke
        parts.append(line(  0,  -rise*0.5, arm, -rise*0.5))   # horizontal tail

    elif tendency == 6:      # Falling then steady  ∟
        parts.append(line(-arm, -rise*0.5,   0,  rise*0.5))   # falling stroke
        parts.append(line(  0,   rise*0.5, arm,  rise*0.5))   # horizontal tail

    return ''.join(parts)


def station_model_svg(d, S=34, wmo_style=False):
    """Full WMO station model SVG. wmo_style param accepted (no-op here)."""
    PAD = S * 1.2
    W   = S * 3 + PAD * 2
    H   = S * 3 + PAD * 2
    cx  = W / 2
    cy  = H / 2
    R   = S * _CR
    fs  = globals().get('FONT_SCALE', max(7, int(S * 0.36)))
    off = R + S * 0.09
    hide_labels = d.get('icao', '').upper() in {'CZPC', 'CWGM'}

    parts = []
    has_cloud = d.get('has_sky_obs', False)
    if has_cloud:
        parts.append(cloud_circle_svg(cx, cy, R, d['oktas']))
    else:
        # No sky sensor — draw black triangle, same centre as circle
        th = R * 1.6
        tx1, ty1 = cx,        cy - th
        tx2, ty2 = cx - th,   cy + th * 0.65
        tx3, ty3 = cx + th,   cy + th * 0.65
        parts.append(f'<polygon points="{tx1:.2f},{ty1:.2f} {tx2:.2f},{ty2:.2f} {tx3:.2f},{ty3:.2f}" fill="black" stroke="none"/>')
    parts.append(wind_barb_svg(cx, cy, R,
                               d['wind_dir'], d['wind_spd'],
                               d.get('wind_gust', 0), S))

    def txt(x, y, text, anchor='end', bold=False, size=None):
        sz = size or fs
        fw = 'bold' if bold else 'normal'
        return (f'<text x="{x:.1f}" y="{y:.1f}" '
                f'text-anchor="{anchor}" dominant-baseline="central" '
                f'font-size="{sz}px" font-weight="{fw}" '
                f'font-family="Courier New,monospace" fill="black" '
                f'paint-order="stroke" stroke="white" '
                f'stroke-width="2" stroke-linejoin="round">'
                f'{text}</text>')

    if not hide_labels:
        if d['temp'] is not None:
            parts.append(txt(cx - off, cy - R * 0.6 - 6, str(d['temp']), bold=False))
        v  = d['vis']
        vs = (str(int(v))  if v is not None and v >= 10    else
              str(int(v))  if v is not None and v % 1 == 0 else
              f'{v:.1f}'   if v is not None                else None)
        wx = ' '.join(x for x in [vs, d['weather'] or None] if x)
        if wx:
            parts.append(txt(cx - off -4, cy, wx, bold=False))
        if d['dew'] is not None:
            parts.append(txt(cx - off, cy + R * 0.6 + 6, str(d['dew'])))
    if d['slp_label']:
        parts.append(txt(cx + off, cy - R * 0.6 - 7, d['slp_label'], anchor='start'))
    tendency = d.get('tendency')
    pressure_change = d.get('pressure_change')
    if not hide_labels and tendency is not None:
        tend_y = cy - R * 0.6 - 7 + S * 0.55
        has_number = tendency != 'steady' and pressure_change is not None
        if has_number:
            pc_str = ('+' if pressure_change > 0 else '-' if pressure_change < 0 else '') + str(abs(pressure_change))
            parts.append(txt(cx + off, tend_y, pc_str, anchor='start'))
            parts.append(pressure_tendency_svg(cx + off, cy, R, tendency, S))
        else:
            # no number — shift symbol left to sit flush with SLP label column
            parts.append(pressure_tendency_svg(cx + off - S * 0.52, cy, R, tendency, S))

    if d['lowest_sig'] and d['lowest_sig']['height'] <= 120:
        _cb = math.ceil(d['lowest_sig']['height'] / 10)
        parts.append(txt(cx, cy + R + fs * 0.9,
                         str(_cb), anchor='middle'))
    _name_y = cy + R + fs * 0.9 + fs * 1.2
    parts.append(txt(cx, _name_y, d['icao'][-3:], anchor='middle'))
    return (f'<svg width="{W:.0f}" height="{H:.0f}" '
            f'viewBox="0 0 {W:.2f} {H:.2f}" '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'style="overflow:visible">'
            + ''.join(parts) + '</svg>'), W, H


def flight_cat_color(d):
    return {'VFR': '#22aa44', 'MVFR': '#2244cc',
            'IFR': '#cc2222', 'LIFR': '#880088'}.get(d.get('flt_cat', ''), '#888888')


# ── DEMO: render 27025KT and a few cardinal directions ────────────────────
print(f'Station model SVG ready  '
      f'(FEATHER_SIDE={FEATHER_SIDE}, FEATHER_ANGLE={FEATHER_ANGLE}°)')
print('Demo wind barbs:')

_demo_records = [
    dict(icao='CYWG', name='Winnipeg',   wind_dir=270, wind_spd=25, wind_gust=0,
         temp=15,  dew=8,   vis=15,  weather='',    slp=1013.2, slp_label='132',
         oktas=4, has_sky_obs=True, clouds=[{'cover':'SCT','height':25,'raw':'SCT025'}],
         lowest_sig=None, ceiling=99999, flt_cat='VFR',   lat=0, lon=0, timestamp='', rh=60,
         tendency='rising',         pressure_change=+28),  # /

    dict(icao='CYYZ', name='Toronto',    wind_dir=0,   wind_spd=20, wind_gust=0,
         temp=2,   dew=-1,  vis=3,   weather='BR',   slp=1001.4, slp_label='014',
         oktas=8, has_sky_obs=True, clouds=[{'cover':'OVC','height':8, 'raw':'OVC008'}],
         lowest_sig={'cover':'OVC','height':8,'raw':'OVC008'}, ceiling=800, flt_cat='IFR',
         lat=0, lon=0, timestamp='', rh=82,
         tendency='falling',        pressure_change=-15),  # \

    dict(icao='CYVR', name='Vancouver',  wind_dir=90,  wind_spd=35, wind_gust=0,
         temp=8,   dew=6,   vis=0.5, weather='-RA',  slp=998.6,  slp_label='986',
         oktas=8, has_sky_obs=True, clouds=[{'cover':'OVC','height':4, 'raw':'OVC004'}],
         lowest_sig={'cover':'OVC','height':4,'raw':'OVC004'}, ceiling=400, flt_cat='LIFR',
         lat=0, lon=0, timestamp='', rh=88,
         tendency='steady',         pressure_change=+2),   # —

    dict(icao='CYQF', name='Red Deer',   wind_dir=180, wind_spd=50, wind_gust=0,
         temp=-5,  dew=-12, vis=15,  weather='',     slp=1020.8, slp_label='208',
         oktas=2, has_sky_obs=True, clouds=[{'cover':'FEW','height':40,'raw':'FEW040'}],
         lowest_sig=None, ceiling=99999, flt_cat='VFR',   lat=0, lon=0, timestamp='', rh=55,
         tendency='rising_falling', pressure_change=+10),  # ∧

    dict(icao='CYYC', name='Calgary',    wind_dir=315, wind_spd=65, wind_gust=0,
         temp=-18, dew=-22, vis=9,   weather='SN',   slp=1008.0, slp_label='080',
         oktas=6, has_sky_obs=True, clouds=[{'cover':'BKN','height':15,'raw':'BKN015'}],
         lowest_sig={'cover':'BKN','height':15,'raw':'BKN015'}, ceiling=1500, flt_cat='MVFR',
         lat=0, lon=0, timestamp='', rh=72,
         tendency='falling_rising', pressure_change=-22),  # V

    dict(icao='CYEG', name='Edmonton',   wind_dir=225, wind_spd=15, wind_gust=0,
         temp=-2,  dew=-8,  vis=15,  weather='',     slp=1015.4, slp_label='154',
         oktas=2, has_sky_obs=True, clouds=[{'cover':'FEW','height':50,'raw':'FEW050'}],
         lowest_sig=None, ceiling=99999, flt_cat='VFR',   lat=0, lon=0, timestamp='', rh=62,
         tendency='rising_steady',  pressure_change=+18),  # ⌐

    dict(icao='CWGM', name='Waterton Gate',      wind_dir=200, wind_spd=10, wind_gust=0,
         temp=-4,  dew=-10, vis=15,  weather='',     slp=1012.1, slp_label='121',
         oktas=4, has_sky_obs=True, clouds=[{'cover':'SCT','height':30,'raw':'SCT030'}],
         lowest_sig=None, ceiling=99999, flt_cat='VFR',   lat=0, lon=0, timestamp='', rh=65,
         tendency='falling_steady', pressure_change=-8),   # ∟
]


_S = 44
_margin = 8
_svg_parts_list = []
for _rec in _demo_records:
    _svg_str, _sw, _sh = station_model_svg(_rec, S=_S)
    _svg_parts_list.append((_svg_str, _sw, _sh, _rec))

_cell_w = int(_svg_parts_list[0][1])
_cell_h = int(_svg_parts_list[0][2])
_label_h = 28
_total_w = len(_svg_parts_list) * (_cell_w + _margin) + _margin
_total_h = _cell_h + _label_h

_svg_parts = [f'<svg width="{_total_w}" height="{_total_h}" '
              f'xmlns="http://www.w3.org/2000/svg" '
              f'style="background:#f8f8f8;font-family:Courier New,monospace">']

for _i, (_svg_str, _sw, _sh, _rec) in enumerate(_svg_parts_list):
    _ox = _margin + _i * (_cell_w + _margin)
    # background cell
    _svg_parts.append(f'<rect x="{_ox}" y="0" width="{_cell_w}" height="{_cell_h}" '
                      f'fill="white" stroke="#ccc" stroke-width="0.8"/>')
    # embed station model shifted to cell origin
    _inner = _svg_str.split('>', 1)[1].rsplit('</svg>', 1)[0]
    _svg_parts.append(f'<g transform="translate({_ox},0)">{_inner}</g>')
    # label: ICAO + wind
    _cx = _ox + _cell_w / 2
    _lbl = f'{_rec["icao"]} {_rec["wind_dir"]:03d}/{_rec["wind_spd"]}kt'
    _svg_parts.append(f'<text x="{_cx:.1f}" y="{_cell_h + 14}" '
                      f'text-anchor="middle" font-size="9" fill="#333">{_lbl}</text>')
    _fc  = flight_cat_color(_rec)
    _svg_parts.append(f'<text x="{_cx:.1f}" y="{_cell_h + 24}" '
                      f'text-anchor="middle" font-size="8" fill="{_fc}" '
                      f'font-weight="bold">{_rec["flt_cat"]}</text>')

_svg_parts.append('</svg>')
_demo_svg = ''.join(_svg_parts)

_legend = """
Station Model Key:
  Top-left    : Temperature (°C)
  Left        : Visibility (SM) + Present Weather
  Bottom-left : Dewpoint (°C)
  Top-right   : SLP last 3 digits (e.g. 132 = 1013.2 hPa, 986 = 998.6 hPa)
  Bottom-right: Pressure change (tenths hPa) + Tendency symbol
                  /  = rising        \\  = falling      —  = steady
                  ∧  = rise then fall  V  = fall then rise
                  ⌐  = rise then steady  ∟ = fall then steady
  Circle      : Sky cover (oktas 0–8), Triangle = no sky sensor
  Barb        : Wind direction (from) and speed (kt)
                  half barb=5kt, full barb=10kt, pennant=50kt

Example:  -10\\  means pressure fell 1.0 hPa over last 3 hours, still falling
          +28/  means pressure rose 2.8 hPa over last 3 hours, still rising
"""

display(HTML('''
<div style="font-family:Courier New,monospace;font-size:12px;background:#f0f4ff;
            border:1px solid #1a4a8a;border-radius:8px;padding:14px 20px;
            max-width:620px;margin:10px 0;color:#1a2030">
  <div style="font-size:14px;font-weight:bold;color:#1a4a8a;border-bottom:1px solid #aac;
              padding-bottom:6px;margin-bottom:10px">📡 Station Model Key</div>
  <table style="border-collapse:collapse;width:100%">
    <tr><td style="color:#888;padding:2px 8px 2px 0;white-space:nowrap">Top-left</td>
        <td>Temperature (°C)</td></tr>
    <tr><td style="color:#888;padding:2px 8px 2px 0;white-space:nowrap">Left</td>
        <td>Visibility (SM) + Present Weather</td></tr>
    <tr><td style="color:#888;padding:2px 8px 2px 0;white-space:nowrap">Bottom-left</td>
        <td>Dewpoint (°C)</td></tr>
    <tr><td style="color:#888;padding:2px 8px 2px 0;white-space:nowrap">Top-right</td>
        <td>SLP last 3 digits &nbsp;<span style="color:#555">e.g. 132 = 1013.2 hPa &nbsp;|&nbsp; 986 = 998.6 hPa</span></td></tr>
    <tr><td style="color:#888;padding:2px 8px 2px 0;white-space:nowrap;vertical-align:top">Bottom-right</td>
        <td>Pressure change (tenths hPa) + Tendency symbol<br>
          <span style="display:inline-block;margin-top:4px">
            <b>/</b> rising &nbsp;
            <b>\\</b> falling &nbsp;
            <b>—</b> steady &nbsp;
            <b>∧</b> rise→fall &nbsp;
            <b>V</b> fall→rise &nbsp;
            <b>⌐</b> rise→steady &nbsp;
            <b>∟</b> fall→steady
          </span>
        </td></tr>
    <tr><td style="color:#888;padding:2px 8px 2px 0;white-space:nowrap">Circle</td>
        <td>Sky cover (oktas 0–8) &nbsp;<span style="color:#555">Triangle = no sky sensor</span></td></tr>
    <tr><td style="color:#888;padding:2px 8px 2px 0;white-space:nowrap">Barb</td>
        <td>Wind direction (from) and speed &nbsp;
            <span style="color:#555">½ barb=5kt &nbsp;| full barb=10kt &nbsp;| pennant=50kt</span></td></tr>
  </table>
  <div style="margin-top:10px;border-top:1px solid #aac;padding-top:8px;color:#444">
    <b>Examples:</b><br>
    <span style="color:#c00">-10 \\</span> &nbsp;→ pressure <b>fell 1.0 hPa</b> over last 3 hrs, still <b>falling</b><br>
    <span style="color:#080">+28 /</span> &nbsp;→ pressure <b>rose 2.8 hPa</b> over last 3 hrs, still <b>rising</b>
  </div>
</div>
'''))

display(SVG(_demo_svg))

# ---- ALBERTA FIRE WEATHER ZONES (self-contained, no file needed) -------
# Replace the entire fire_zones_html block in Cell 9 with this.

# ── Fetch Alberta Fire Weather Forecast Zones KML from repo → GeoJSON ────────
# Raw KML committed at a pinned commit in the SfcMap repo.
# Parsed with the stdlib xml.etree.ElementTree — no extra dependencies.
import xml.etree.ElementTree as _ET

_KML_URL = (
    'https://github.com/ngsmetadvisor/SfcMap/raw/920cf65213038f03b6c927f218f76297c5c619c6/Alberta_Fire_Weather_Forecast_Zones.kml'
)
_fire_zones_geojson_str = json.dumps({"type": "FeatureCollection", "features": []})

def _kml_coords_to_ring(coord_text):
    ring = []
    for token in coord_text.strip().split():
        parts = token.split(',')
        if len(parts) >= 2:
            try:
                ring.append([float(parts[0]), float(parts[1])])
            except ValueError:
                pass
    return ring

def _kml_to_geojson(kml_bytes):
    root = _ET.fromstring(kml_bytes)

    # ── FIX 1: auto-detect namespace ──────────────────────────────────────────
    ns_uri = ''
    if root.tag.startswith('{'):
        ns_uri = root.tag.split('}')[0][1:]

    def tag(name):
        return f'{{{ns_uri}}}{name}' if ns_uri else name

    features = []
    for pm in root.iter(tag('Placemark')):
        name = pm.findtext(tag('name')) or 'Fire Zone'
        polygons = list(pm.iter(tag('Polygon')))
        if not polygons:
            continue

        rings_list = []
        for poly in polygons:
            outer = poly.find(
                f'.//{tag("outerBoundaryIs")}/{tag("LinearRing")}/{tag("coordinates")}'
            )
            if outer is None or not outer.text:
                continue
            ring = _kml_coords_to_ring(outer.text)
            if len(ring) < 3:
                continue
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            rings_list.append(ring)

        if not rings_list:
            continue

        if len(rings_list) == 1:
            geometry = {"type": "Polygon", "coordinates": [rings_list[0]]}
        else:
            # ── FIX 2: correct MultiPolygon structure ─────────────────────────
            # MultiPolygon: [ polygon, polygon, ... ]
            # each polygon:  [ outerRing, ...holes ]
            geometry = {
                "type": "MultiPolygon",
                "coordinates": [[ring] for ring in rings_list]
            }

        features.append({
            "type": "Feature",
            "properties": {"name": name.strip()},
            "geometry": geometry,
        })
    return {"type": "FeatureCollection", "features": features}

try:
    print('Fetching Alberta Fire Weather Forecast Zones KML from repo...')
    _fz_resp = requests.get(_KML_URL, timeout=30)
    _fz_resp.raise_for_status()
    _fz_geojson = _kml_to_geojson(_fz_resp.content)
    _fire_zones_geojson_str = json.dumps(_fz_geojson)
    _fz_count = len(_fz_geojson.get('features', []))
    print(f'✓ Alberta Fire Weather Forecast Zones loaded: {_fz_count} zones')
    # ── DEBUG: print first feature so you can verify the structure ────────────
    if _fz_geojson['features']:
        print(f'  First feature: {_fz_geojson["features"][0]["properties"]["name"]}')
        print(f'  Geometry type: {_fz_geojson["features"][0]["geometry"]["type"]}')
except Exception as _fz_err:
    print(f'⚠ Fire zones fetch/parse failed ({_fz_err}) — map will load without fire zone layer')


fire_zones_html = (
    '<script>\n'
    'var _FIRE_ZONES_GEOJSON = ' + _fire_zones_geojson_str + ';\n'
    '(function() {\n'
    '  var _attempts = 0;\n'
    '  function loadFireZones() {\n'
    # ── FIX 3: cap retries + validate map object ─────────────────────────────
    '    _attempts++;\n'
    '    if (_attempts > 40) {\n'
    '      console.warn("Fire zones: map not found after 40 attempts. Available window keys:", Object.keys(window).filter(function(k){return k.startsWith("map_");}));\n'
    '      return;\n'
    '    }\n'
    '    var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    if (!keys.length) { setTimeout(loadFireZones, 300); return; }\n'
    '    var MAP = window[keys[0]];\n'
    # ── FIX 4: confirm it is actually a Leaflet map ──────────────────────────
    '    if (!MAP || typeof MAP.addLayer !== "function") { setTimeout(loadFireZones, 300); return; }\n'
    '    console.log("Fire zones: attaching to", keys[0], "| features:", _FIRE_ZONES_GEOJSON.features.length);\n'
    '    if (!_FIRE_ZONES_GEOJSON.features.length) {\n'
    '      console.warn("Fire zones: GeoJSON has 0 features — check KML parse"); return;\n'
    '    }\n'
    '    var fireLayer = L.geoJSON(_FIRE_ZONES_GEOJSON, {\n'
    '      style: function() {\n'
    '        return {\n'
    '          color: "#cc0000",\n'
    '          weight: 1.8,\n'
    '          opacity: 0.85,\n'
    '          fillColor: "#ff9933",\n'
    '          fillOpacity: 0.00,\n'
    '          dashArray: null\n'
    '        };\n'
    '      },\n'
    '      onEachFeature: function(feature, layer) {\n'
    '        var name = (feature.properties && feature.properties.name) || "Fire Zone";\n'
    '        layer.bindTooltip(name, {sticky: true, opacity: 0.9});\n'
    '        layer.bindPopup(\n'
    '          \'<div style="font-family:Courier New,monospace;font-size:12px;">\'\n'
    '          + \'<b style="color:#cc4400">\' + name + \'</b><br>\'\n'
    '          + \'Alberta Fire Weather Forecast Zone\'\n'
    '          + \'</div>\'\n'
    '        );\n'
    '      }\n'
    '    });\n'
    '    var _fireVisible = true;\n'
    '    var btn = document.getElementById("btn-fire-zones");\n'
    '    if (btn) {\n'
    '      btn.onclick = function() {\n'
    '        _fireVisible = !_fireVisible;\n'
    '        if (_fireVisible) { fireLayer.addTo(MAP); btn.style.background = "#cc4400"; }\n'
    '        else { MAP.removeLayer(fireLayer); btn.style.background = "#b0b8c8"; }\n'
    '      };\n'
    '      btn.style.background = "#cc4400";\n'
    '    }\n'
    '    fireLayer.addTo(MAP);\n'
    '  }\n'
    '  if (document.readyState === "complete") { setTimeout(loadFireZones, 800); }\n'
    '  else { window.addEventListener("load", function(){ setTimeout(loadFireZones, 800); }); }\n'
    '})();\n'
    '</script>\n'
    '<style>#btn-fire-zones { transition: background 0.2s; }</style>\n'
)

print('Alberta Fire Zone XML imported')

# ── Cell 3 . Load station list from orangecore.net ────────────
import csv, io, math as _math

def load_stations(url, coverage='standard'):
    # Use a local copy if available — avoids network dependency on the runner
    _local = 'AP_location.csv'
    if _os.path.exists(_local):
        print(f'Using local station list: {_local}')
        with open(_local, encoding='utf-8') as _f:
            _text = _f.read()
    else:
        print(f'Fetching station list from {url}')
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        _text = r.text
        # Cache it locally for next time
        with open(_local, 'w', encoding='utf-8') as _f:
            _f.write(_text)
    reader = csv.DictReader(io.StringIO(_text))
    stations = {}
    for row in reader:
        icao = row.get('Code','').strip()
        if not icao: continue

        if coverage == 'chart':
            chart_keys = [k for k in row.keys() if k.strip().lower() == 'chart']
            chart_val  = row.get(chart_keys[0], '').strip() if chart_keys else ''
            if not chart_val:
                continue
        else:
            tier_map = {'essential': 1, 'standard': 2, 'all': 3}
            max_tier = tier_map.get(coverage, 2)
            tier = (1 if row.get('ESSENTIAL','').strip() else
                    2 if row.get('STANDARD','').strip()  else 3)
            if tier > max_tier:
                continue

        try:
            stations[icao] = {
                'icao': icao,
                'name': row.get('Name','').strip(),
                'lat':  float(row['Latitude']),
                'lon':  float(row['Longitude']),
                'tier': 0 if coverage == 'chart' else tier,
                'source': 'metar',
            }
        except (ValueError, KeyError):
            pass

    return stations

STATIONS = load_stations(CSV_URL, COVERAGE)
print(f'✓ Loaded {len(STATIONS)} stations ({COVERAGE} tier)')

lats = [s['lat'] for s in STATIONS.values()]
lons = [s['lon'] for s in STATIONS.values()]
print(f'  Lat range: {min(lats):.1f}°N - {max(lats):.1f}°N')
print(f'  Lon range: {min(lons):.1f}°E - {max(lons):.1f}°E')


# ── NEW: Register EC model virtual stations into STATIONS ─────────────────────
# Injected here so parse_metar_line() and ALL downstream cells see them
# as ordinary stations, indistinguishable from real ones except source='ecmodel'.

EC_LONGITUDE  = -139.7                    # 139.7°W — PAYA (Yakutat) meridian
EC_LATITUDES  = list(range(49, 0, -10))  # [49, 39, 29, 19, 9] south to equator
OPENMETEO_URL = 'https://api.open-meteo.com/v1/forecast'

def ec_icao(lat, lon):
    """Build a compact ICAO-slot-length ID: e.g. ECMLN61W150"""
    return (f"ECM{'N' if lat>=0 else 'S'}{abs(lat):02d}"
            f"{'E' if lon>=0 else 'W'}{abs(int(lon)):03d}")

for _lat in EC_LATITUDES:
    _id = ec_icao(_lat, EC_LONGITUDE)
    STATIONS[_id] = {
        'icao':   _id,
        'name':   f'EC Model {_lat:+d}N {abs(EC_LONGITUDE):.0f}W',
        'lat':    float(_lat),
        'lon':    float(EC_LONGITUDE),
        'tier':   0,
        'source': 'ecmodel',
    }

print(f'  + {len(EC_LATITUDES)} EC model virtual stations registered')
print(f'  Total STATIONS: {len(STATIONS)}')


# ── Cell 3b . Fetch & parse EC model data from Open-Meteo ─────────────────────
# Produces ec_metar_records[] with the IDENTICAL schema as parse_metar_line().
# Appended to metar_records[] at the END of Cell 5 — before Cell 5b runs.

from datetime import datetime, timezone as _tz



def _ec_rh(temp, dew):
    if temp is None or dew is None: return None
    a, b = 17.625, 243.04
    rh = round(100 * _math.exp((a*dew/(b+dew)) - (a*temp/(b+temp))))
    return max(0, min(100, rh))

def _ec_slp_label(slp):
    return '' if slp is None else f'SLP{int(round(slp * 10)) % 1000:03d}'

def _ec_vis(prec):
    if prec is None or prec < 0.1: return 10.0
    if prec < 2.5:  return 5.0
    if prec < 7.6:  return 2.0
    return 0.5

def _ec_wx(prec):
    if prec is None or prec < 0.1: return ''
    if prec < 2.5: return '-RA'
    if prec < 7.6: return 'RA'
    return '+RA'

def _ec_cat(vis, ceil):
    if ceil < 500  or vis < 1: return 'LIFR'
    if ceil < 1000 or vis < 3: return 'IFR'
    if ceil < 3000 or vis < 5: return 'MVFR'
    return 'VFR'

def _ec_tfmt(c):
    if c is None: return '//'
    i = int(round(c))
    return f'M{abs(i):02d}' if i < 0 else f'{i:02d}'

# ── fetch one grid point ──────────────────────────────────────────────────────
def _fetch_ec(lat, lon, past_days=2, forecast_days=1):
    r = requests.get(OPENMETEO_URL, params={
        'latitude': lat, 'longitude': lon,
        'hourly': ('temperature_2m,precipitation,pressure_msl,'
                   'wind_speed_10m,wind_direction_10m,wind_gusts_10m,dew_point_2m'),
        'models': 'ecmwf_ifs',
        'past_days': past_days, 'forecast_days': forecast_days,
        'wind_speed_unit': 'kn', 'timezone': 'UTC',
    }, timeout=20)
    r.raise_for_status()
    return r.json()

# ── parse one response → list of METAR-schema dicts ──────────────────────────
def _parse_ec(lat, lon, data):
    icao   = ec_icao(lat, lon)
    st     = STATIONS[icao]
    hourly = data.get('hourly', {})
    times  = hourly.get('time', [])

    def col(k): return hourly.get(k, [])
    T  = col('temperature_2m');   D  = col('dew_point_2m')
    SL = col('pressure_msl')
    WD = col('wind_direction_10m'); WS = col('wind_speed_10m')
    WG = col('wind_gusts_10m');   PR = col('precipitation')

    records = []
    for i, iso in enumerate(times):
        def g(lst): return lst[i] if i < len(lst) else None
        temp=g(T); dew=g(D); slp=g(SL)
        wdir=g(WD); wspd=g(WS); wgst=g(WG); prec=g(PR)

        # Timestamp → DDHHmmZ (model data is always on the hour)
        try:
            dt = datetime.fromisoformat(iso).replace(tzinfo=_tz.utc)
            ts = dt.strftime('%d%H00Z')
        except Exception:
            ts = '//////Z'

        sky         = 'SKC'
        oktas       = 0
        clouds_list = []
        ceiling     = 99999
        lowest_sig  = None

        vis     = _ec_vis(prec)
        wx      = _ec_wx(prec)
        flt_cat = _ec_cat(vis, ceiling)
        rh      = _ec_rh(temp, dew)
        slp_lbl = _ec_slp_label(slp)

        if wdir is not None and wspd is not None:
            wd, ws = int(round(wdir)), int(round(wspd))
            wg     = int(round(wgst)) if wgst else 0
            gust   = f'G{wg:02d}' if wg > ws + 5 else ''
            wind_g = f'{wd:03d}{ws:02d}{gust}KT'
            wind_gust_out = wg if (wgst and wgst > wspd + 5) else 0
        else:
            wind_g = '/////KT'; wd = ws = wg = None; wind_gust_out = 0

        vis_str   = f'{int(vis)}SM' if vis == int(vis) else f'{vis:.1f}SM'
        metar_str = ' '.join(p for p in [
            'METAR', icao, ts, 'AUTO', wind_g, vis_str, wx,
            sky, f'{_ec_tfmt(temp)}/{_ec_tfmt(dew)}', slp_lbl,
            'RMK ECMWF_IFS',
        ] if p)

        records.append(dict(
            icao=icao, name=st['name'],
            lat=lat, lon=lon,
            source='ecmodel',
            timestamp=ts,
            wind_dir=wdir, wind_spd=wspd, wind_gust=wind_gust_out,
            vis=vis, temp=temp, dew=dew, rh=rh,
            slp=slp, slp_label=slp_lbl,
            has_sky_obs=False, oktas=oktas,
            clouds=clouds_list, lowest_sig=lowest_sig, ceiling=ceiling,
            weather=wx, flt_cat=flt_cat,
            tendency=None, pressure_change=None,
            metar_str=metar_str,
        ))
    return records

# ── fetch loop ────────────────────────────────────────────────────────────────
ec_metar_records = []
ec_fetch_errors  = []

print(f'Fetching {len(EC_LATITUDES)} EC model stations along {abs(EC_LONGITUDE):.0f}°W transect...')
for _lat in EC_LATITUDES:
    _id = ec_icao(_lat, EC_LONGITUDE)
    print(f'  {_id}  ({_lat:+03d}°N) … ', end='')
    try:
        _recs = _parse_ec(_lat, EC_LONGITUDE, _fetch_ec(_lat, EC_LONGITUDE))
        ec_metar_records.extend(_recs)
        print(f'✓  {len(_recs)} hourly obs')
    except Exception as _e:
        print(f'✗  {_e}')
        ec_fetch_errors.append(_id)

print(f'\n✓ EC model records ready: {len(ec_metar_records)}')
if ec_fetch_errors:
    print(f'  ✗ Failed: {ec_fetch_errors}')


# ── Cell 4 . Fetch live METARs from aviationweather.gov ───────
import concurrent.futures, time

EXPECTED_HOURS = [0, 6, 12, 18]

def fetch_chunk(codes, hours=12, retries=3, backoff=2):
    for attempt in range(retries):
        try:
            params = {'ids': ','.join(codes), 'format': 'raw',
                      'hours': hours, 'mostRecent': 'false'}
            r = requests.get(METAR_API, params=params, timeout=30)
            if r.ok and r.text.strip():
                return r.text, []
            time.sleep(backoff * (attempt + 1))
        except Exception:
            time.sleep(backoff * (attempt + 1))
    return '', codes

def fetch_all_metars(station_codes, chunk_size=25, max_workers=6, hours=12):
    chunks = [station_codes[i:i+chunk_size]
              for i in range(0, len(station_codes), chunk_size)]
    chunk_lines = ''.join(
        f'<div style="font-family:monospace;font-size:11px;color:#555;margin:1px 0">'
        f'Chunk {i+1}: {", ".join(c)}</div>'
        for i, c in enumerate(chunks)
    )
    display(HTML(f'''
    <details style="margin:6px 0;font-family:monospace;font-size:12px;">
      <summary style="cursor:pointer;color:#1a4a8a;font-weight:bold;">
        Fetching {len(station_codes)} stations in {len(chunks)} chunks
        ({max_workers} workers, up to 3 retries) — click to expand
      </summary>
      <div style="margin-top:6px;padding:8px;background:#f8f8f8;
                  border:1px solid #ddd;border-radius:4px;max-height:200px;overflow-y:auto;">
        {chunk_lines}
      </div>
    </details>
    '''))

    raw_parts = []; failed_codes = []; done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_chunk, c, hours): c for c in chunks}
        for fut in concurrent.futures.as_completed(futures):
            text, failed = fut.result()
            if text:  raw_parts.append(text)
            if failed: failed_codes.extend(failed)
            done += len(futures[fut])
            print(f'  {done}/{len(station_codes)} ({int(done/len(station_codes)*100)}%)', end='\r')
    print()

    joined = '\n'.join(raw_parts)
    seen_icaos = set()
    for line in joined.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            icao = parts[1] if parts[0] in ('METAR','SPECI') else parts[0]
            seen_icaos.add(icao)

    silent_missing = [s for s in station_codes
                      if s not in seen_icaos and s not in failed_codes]
    if silent_missing:
        print(f'  ↻ Pass 2: retrying {len(silent_missing)} silent-missing stations...')
        retry_chunks = [silent_missing[i:i+chunk_size]
                        for i in range(0, len(silent_missing), chunk_size)]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures2 = {ex.submit(fetch_chunk, c, hours, 2, 1): c for c in retry_chunks}
                for fut in concurrent.futures.as_completed(futures2, timeout=10):
                    text, failed = fut.result()
                    if text:  raw_parts.append(text)
                    if failed: failed_codes.extend(failed)
            print('  ↻ Pass 2 done.')
        except concurrent.futures.TimeoutError:
            print(f'  ↻ Pass 2 timed out — skipping {len(silent_missing)} stations')

    return '\n'.join(raw_parts), failed_codes


# Only send real station codes to aviationweather.gov (not EC virtual stations)
real_codes = [c for c in STATIONS if STATIONS[c].get('source') != 'ecmodel']
raw_metar_text, failed_chunks = fetch_all_metars(real_codes, hours=12)
line_count = sum(1 for l in raw_metar_text.splitlines() if l.strip())
print(f'✓ Fetched {line_count} raw METAR lines')

returned_icaos = set()
for line in raw_metar_text.splitlines():
    parts = line.strip().split()
    if len(parts) >= 2:
        icao = parts[1] if parts[0] in ('METAR', 'SPECI') else parts[0]
        returned_icaos.add(icao)

no_data = [s for s in real_codes if s not in returned_icaos]

warnings = []
if failed_chunks:
    warnings.append(f"CHUNK FETCH FAILED — {len(failed_chunks)} stations lost:<br>"
                    + "&nbsp;&nbsp;" + "&nbsp;&nbsp;".join(failed_chunks))
if no_data:
    warnings.append(f"NO DATA RETURNED for {len(no_data)} stations:<br>"
                    + "&nbsp;&nbsp;" + "&nbsp;&nbsp;".join(no_data))

if warnings:
    display(HTML(f'''
    <div style="background:#ffb3c6;border:4px solid #cc0000;border-radius:10px;
                padding:28px 32px;margin:16px 0;">
      <div style="color:#cc0000;font-size:32px;font-family:monospace;margin-bottom:16px;">⚠ METAR FETCH WARNING</div>
      {"".join(f'<div style="color:#990000;font-size:18px;font-family:monospace;margin-bottom:12px;line-height:1.6;">{w}</div>' for w in warnings)}
    </div>'''))
else:
    display(HTML('''
    <div style="background:#b6f5c8;border:4px solid #1a7a3a;border-radius:10px;
                padding:28px 32px;margin:16px 0;">
      <div style="color:#145c2c;font-size:32px;font-family:monospace;margin-bottom:10px;">✔ ALL STATIONS FETCHED SUCCESSFULLY</div>
      <div style="color:#1a7a3a;font-size:20px;font-family:monospace;">All stations returned data — no warnings.</div>
    </div>'''))


# ── Cell 5 . Parse METAR fields ───────────────────────────────

def parse_metar_line(line, stations):
    '''Parse one METAR line → dict or None'''
    parts = line.strip().split()
    if len(parts) < 5: return None
    idx = 0
    if parts[0] == 'SPECI': return None
    if parts[0] == 'METAR': idx = 1
    if idx >= len(parts): return None
    icao = parts[idx]
    if icao not in stations: return None
    st = stations[icao]

    ts_raw = parts[idx+1] if idx+1 < len(parts) else ''
    if not re.match(r'^\d{6}Z$', ts_raw): return None
    day, hour, minute = int(ts_raw[0:2]), int(ts_raw[2:4]), int(ts_raw[4:6])
    if minute >= 35:
        hour = (hour + 1) % 24; minute = 0
    elif minute <= 25:
        minute = 0
    else:
        return None
    timestamp = f'{day:02d}{hour:02d}00Z'

    rest = parts[idx+2:]
    rest = [p for p in rest if p not in ('MISG', 'MSIG')]

    wind_dir = wind_spd = wind_gust = None
    for p in rest:
        m = re.match(r'^(\d{3})(\d{2,3})(?:G(\d{2,3}))?KT$', p)
        if m:
            wind_dir, wind_spd = int(m[1]), int(m[2])
            wind_gust = int(m[3]) if m[3] else 0
            break
        if re.match(r'^00000KT$', p): wind_dir=0; wind_spd=0; wind_gust=0; break

    vis = None
    for i, p in enumerate(rest):
        if p.endswith('SM'):
            whole = int(rest[i-1]) if i > 0 and rest[i-1].isdigit() else 0
            frac_str = p[:-2].lstrip('M')
            if '/' in frac_str:
                try:
                    n, d = frac_str.split('/')
                    vis = whole + int(n) / int(d)
                except (ValueError, ZeroDivisionError):
                    vis = 0.0
            else:
                try: vis = whole + float(frac_str) if frac_str else float(whole)
                except: vis = None
            break

    cloud_re = re.compile(r'^(FEW|SCT|BKN|OVC|VV)(\d{3})')
    clouds = []
    for p in rest:
        m = cloud_re.match(p)
        if m: clouds.append({'cover': m[1], 'height': int(m[2]), 'raw': p})
    clouds.sort(key=lambda c: c['height'])
    clr = any(p in ('CLR', 'SKC', 'CAVOK') for p in rest)
    has_sky_obs = clr or bool(clouds)
    cover_rank = {'CLR':0,'SKC':0,'FEW':2,'SCT':4,'BKN':6,'OVC':8,'VV':9}
    oktas = 0 if (clr or not clouds) else max(cover_rank.get(c['cover'], 0) for c in
                  ([c for c in clouds if c['cover'] in ('BKN','OVC','VV')] or clouds))
    sig_clouds = [c for c in clouds if c['cover'] in ('BKN','OVC','VV')]
    ceiling    = sig_clouds[0]['height'] * 100 if sig_clouds else 99999
    lowest_sig = sig_clouds[0] if sig_clouds else None

    temp = dew = None
    for p in rest:
        m = re.match(r'^(M?\d{1,2})/(M?\d{1,2})$', p)
        if m:
            def td(s): return -(int(s[1:])) if s.startswith('M') else int(s)
            temp, dew = td(m[1]), td(m[2])
            break

    slp = None
    for p in rest:
        m = re.match(r'^SLP(\d{3})$', p)
        if m:
            v = int(m[1])
            slp = (900 + v/10) if v >= 500 else (1000 + v/10)
            break

    wx_re = re.compile(
        r'^[+-]?(FZ|SH|BL|TS|MI|PR|BC|DR)?'
        r'(DZ|RA|SN|SG|IC|PL|GR|GS|UP|FG|BR|HZ|FU|VA|DU|SA|SQ|PO|FC|SS|DS){1,3}$'
    )
    wx_parts = [p for p in rest
                if wx_re.match(p)
                and not re.match(r'^(RMK|SLP|AUTO|COR|AO\d)', p)]
    weather = ' '.join(wx_parts)

    rh = None
    if temp is not None and dew is not None:
        a, b = 17.625, 243.04
        rh = round(100 * np.exp((a*dew/(b+dew)) - (a*temp/(b+temp))))
        rh = max(0, min(100, rh))

    fc_vis = vis if vis is not None else 99
    if   ceiling < 500  or fc_vis < 1: flt_cat = 'LIFR'
    elif ceiling < 1000 or fc_vis < 3: flt_cat = 'IFR'
    elif ceiling < 3000 or fc_vis < 5: flt_cat = 'MVFR'
    else:                               flt_cat = 'VFR'

    slp_label = f'{int(round(slp*10))%1000:03d}' if slp else ''

    return dict(
        icao=icao, name=st['name'],
        lat=st['lat'], lon=st['lon'],
        source='metar',
        timestamp=timestamp,
        wind_dir=wind_dir, wind_spd=wind_spd, wind_gust=wind_gust,
        vis=vis, temp=temp, dew=dew, rh=rh, slp=slp, slp_label=slp_label,
        has_sky_obs=has_sky_obs, oktas=oktas, clouds=clouds, lowest_sig=lowest_sig,
        ceiling=ceiling, weather=weather, flt_cat=flt_cat,
        tendency=None, pressure_change=None,
        metar_str=line.strip(),
    )


def parse_all(text, stations):
    results = []
    seen = set()
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        if not re.match(r'^(METAR |SPECI |[A-Z]{4} \d{6}Z)', line): continue
        if line.startswith(('MISG', 'MSIG', 'SIGMET', 'AIRMET', 'PIREP', 'ATIS')): continue
        d = parse_metar_line(line, stations)
        if d:
            key = (d['icao'], d['timestamp'])
            if key not in seen:
                seen.add(key)
                results.append(d)
    return results


metar_records = parse_all(raw_metar_text, STATIONS)

# ── CYMJ/CYYN mutual exclusion ────────────────────────────────────────────────
_cymj_times = set(d['timestamp'] for d in metar_records if d['icao'] == 'CYMJ')
if _cymj_times:
    _before = len(metar_records)
    metar_records = [d for d in metar_records if not (d['icao'] == 'CYYN')]
    print(f'  CYMJ present — removed CYYN ({_before - len(metar_records)} records dropped)')
else:
    print(f'  CYMJ not available — keeping CYYN')
# ── CZPC/CYQL mutual exclusion ────────────────────────────────────────────────
_czpc_times = set(d['timestamp'] for d in metar_records if d['icao'] == 'CZPC')
if _czpc_times:
    _before = len(metar_records)
    metar_records = [d for d in metar_records if not (d['icao'] == 'CYQL')]
    print(f'  CZPC present — removed CYQL ({_before - len(metar_records)} records dropped)')
else:
    print(f'  CZPC not available — keeping CYQL')
# ──────────────────────────────────────────────────────────────────────────────






# ── Merge EC model records ────────────────────────────────────────────────────
# Done here, BEFORE the timestep check and BEFORE Cell 5b, so tendency is
# computed identically for both real and model data.
_n_real = len(metar_records)
_metar_timestamps = set(d['timestamp'] for d in metar_records)
ec_metar_records  = [d for d in ec_metar_records if d['timestamp'] in _metar_timestamps]
metar_records     = metar_records + ec_metar_records
print(f'✓ Merged: {_n_real} real METARs + {len(ec_metar_records)} EC obs = {len(metar_records)} total')

# ── Warn on missing timesteps ─────────────────────────────────────────────────
from collections import defaultdict

all_timestamps = sorted(set(d['timestamp'] for d in metar_records))
station_times  = defaultdict(set)
for d in metar_records:
    station_times[d['icao']].add(d['timestamp'])

missing = {}
for icao in station_times:
    gaps = [ts for ts in all_timestamps if ts not in station_times[icao]]
    if gaps:
        missing[icao] = gaps

total_missing = sum(len(v) for v in missing.values())

slp_count  = sum(1 for d in metar_records if d['slp'])
wind_count = sum(1 for d in metar_records if d['wind_dir'] is not None)
temp_count = sum(1 for d in metar_records if d['temp'] is not None)

if missing:
    rows = ''.join(
        f'<tr><td style="padding:3px 14px;color:#5a3a00;font-family:monospace;">{icao}</td>'
        f'<td style="padding:3px 14px;color:#7a5000;font-family:monospace;text-align:center;">{len(gaps)}</td>'
        f'<td style="padding:3px 14px;color:#7a5000;font-family:monospace;">{", ".join(gaps)}</td></tr>'
        for icao, gaps in sorted(missing.items())
    )
    display(HTML(f'''
    <div style="background:#fff3b0;border:4px solid #e6a800;border-radius:10px;padding:24px 28px;margin:16px 0;">
      <div style="color:#b37700;font-size:28px;font-family:monospace;margin-bottom:6px;">⚠ MISSING TIMESTEPS DETECTED</div>
      <div style="color:#7a5000;font-size:16px;font-family:monospace;margin-bottom:14px;">
        {len(missing)} stations affected &nbsp;|&nbsp; {total_missing} total missing timesteps
        &nbsp;|&nbsp; {len(all_timestamps)} timesteps expected per station
      </div>
      <button onclick="var t=document.getElementById('missing-table');var b=document.getElementById('missing-btn');
          if(t.style.display==='none'){{t.style.display='block';b.textContent='▲ Collapse';}}
          else{{t.style.display='none';b.textContent='▼ Show affected stations';}}"
        id="missing-btn" style="font-family:monospace;font-size:13px;padding:4px 14px;
        background:#ffe066;border:1px solid #e6a800;border-radius:4px;color:#5a3a00;cursor:pointer;margin-bottom:10px;">
        ▼ Show affected stations
      </button>
      <div id="missing-table" style="display:none;">
        <table style="border-collapse:collapse;width:100%;">
          <tr style="background:#ffe066;">
            <th style="padding:4px 14px;text-align:left;color:#5a3a00;font-family:monospace;">ICAO</th>
            <th style="padding:4px 14px;text-align:center;color:#5a3a00;font-family:monospace;"># Missing</th>
            <th style="padding:4px 14px;text-align:left;color:#5a3a00;font-family:monospace;">Missing Timesteps</th>
          </tr>{rows}
        </table>
      </div>
    </div>'''))
    _all_stations  = sorted(set(d['icao'] for d in metar_records))
    _good_stations = sorted(set(d['icao'] for d in metar_records) - set(missing.keys()))
    # Build latest record per station for tooltip
    _latest = {}
    for d in metar_records:
        if d['icao'] not in _latest or d['timestamp'] > _latest[d['icao']]['timestamp']:
            _latest[d['icao']] = d

    def _station_badge(icao):
        d = _latest.get(icao, {})
        has_gap  = icao in missing
        bg       = '#fff3b0' if has_gap else '#e6faf0'
        bdr      = '#e6a800' if has_gap else '#1a7a3a'
        clr      = '#7a5000' if has_gap else '#145c2c'
        temp_str = f"{d.get('temp','—')}°C" if d.get('temp') is not None else '—'
        dew_str  = f"{d.get('dew','—')}°C"  if d.get('dew')  is not None else '—'
        slp_str  = f"{d.get('slp','—')} hPa" if d.get('slp') is not None else '—'
        wdir     = d.get('wind_dir')
        wspd     = d.get('wind_spd')
        wind_str = f"{wdir}°/{wspd}kt" if wdir is not None and wspd is not None else '—'
        cat      = d.get('flt_cat', '—')
        ts       = d.get('timestamp', '—')
        src      = d.get('source', '—')
        gap_str  = f"⚠ missing: {', '.join(missing[icao])}" if has_gap else '✔ complete'
        detail_id = f'stn-detail-{icao}'
        popup_bdr = '#a85c00' if has_gap else '#1a7a3a'
        popup_gap_clr = '#a85c00' if has_gap else '#1a7a3a'
        # All METAR lines for this station, sorted by timestamp
        all_lines = [
            r for r in metar_records if r['icao'] == icao
        ]
        all_lines.sort(key=lambda r: r['timestamp'])
        metar_rows = ''.join(
            f'<tr style="border-bottom:1px solid #eee;">'
            f'<td style="padding:2px 8px;color:#555;white-space:nowrap;">{r["timestamp"]}</td>'
            f'<td style="padding:2px 8px;font-family:monospace;font-size:10px;'
            f'color:#1a2030;white-space:nowrap;">{r.get("metar_str","—")}</td>'
            f'</tr>'
            for r in all_lines
        )

        return (
            f'<span style="display:inline-block;position:relative;margin:2px;">'
            f'<span onclick="'
            f'var p=document.getElementById(\'{detail_id}\');'
            f'document.querySelectorAll(\'.stn-detail-popup\').forEach(function(x){{if(x.id!==\'{detail_id}\')x.style.display=\'none\';}});'
            f'p.style.display=p.style.display===\'none\'?\'block\':\'none\';" '
            f'style="font-family:monospace;font-size:11px;color:{clr};cursor:pointer;'
            f'background:{bg};border:1px solid {bdr};border-radius:3px;'
            f'padding:1px 6px;display:inline-block;">{icao}</span>'
            f'<div id="{detail_id}" class="stn-detail-popup" '
            f'style="display:none;position:absolute;top:20px;left:0;z-index:9999;'
            f'background:#fff;border:2px solid {popup_bdr};border-radius:8px;padding:12px 16px;'
            f'font-family:monospace;font-size:12px;color:#1a2030;'
            f'box-shadow:0 4px 16px rgba(0,0,0,0.25);min-width:420px;max-width:700px;">'
            f'<b style="font-size:13px;color:#1a2030;">{icao}</b> '
            f'<span style="color:#888;font-size:10px;">{d.get("name","")}</span> '
            f'<span style="color:#888;font-size:10px;">· {src} · {len(all_lines)} obs</span>'
            f'<hr style="margin:4px 0;border:none;border-top:1px solid #ccc;">'
            f'<span style="color:{popup_gap_clr};font-size:10px;">{gap_str}</span>'
            f'<hr style="margin:4px 0;border:none;border-top:1px solid #eee;">'
            f'<div style="max-height:300px;overflow-y:auto;">'
            f'<table style="border-collapse:collapse;width:100%;font-size:10px;">'
            f'<tr style="background:#f0f4f8;"><th style="padding:2px 8px;text-align:left;">Time</th>'
            f'<th style="padding:2px 8px;text-align:left;">METAR</th></tr>'
            f'{metar_rows}'
            f'</table>'
            f'</div>'
            f'<button onclick="document.getElementById(\'{detail_id}\').style.display=\'none\';event.stopPropagation();" '
            f'style="margin-top:8px;font-size:10px;padding:2px 10px;cursor:pointer;'
            f'border:1px solid #aaa;border-radius:3px;background:#f0f0f0;">✕ close</button>'
            f'</div>'
            f'</span>'
        )

    _good_rows = ''.join(_station_badge(icao) for icao in _all_stations)
    display(HTML(f'''
    <div style="background:#b6f5c8;border:4px solid #1a7a3a;border-radius:10px;padding:24px 28px;margin:16px 0;overflow:visible;">
      <div style="color:#145c2c;font-size:24px;font-family:monospace;margin-bottom:8px;">✔ {len(_all_stations)} STATIONS WITH DATA &nbsp;|&nbsp; {len(_good_stations)} COMPLETE</div>
      <div style="color:#1a7a3a;font-size:13px;font-family:monospace;margin-bottom:10px;">
        {len(all_timestamps)} timesteps &nbsp;|&nbsp; {len(metar_records)} total records &nbsp;|&nbsp;
        SLP: {slp_count} &nbsp; Wind: {wind_count} &nbsp; Temp: {temp_count}
      </div>
      <div style="line-height:2.2;overflow:visible;">{_good_rows}</div>
    </div>'''))
else:
    display(HTML(f'''
    <div style="background:#b6f5c8;border:4px solid #1a7a3a;border-radius:10px;padding:24px 28px;margin:16px 0;">
      <div style="color:#145c2c;font-size:28px;font-family:monospace;margin-bottom:8px;">✔ ALL STATIONS HAVE COMPLETE TIMESTEPS</div>
      <div style="color:#1a7a3a;font-size:18px;font-family:monospace;">
        {len(all_timestamps)} timesteps &nbsp;|&nbsp; {len(metar_records)} total records &nbsp;|&nbsp;
        SLP: {slp_count} &nbsp; Wind: {wind_count} &nbsp; Temp: {temp_count}
      </div>
    </div>'''))


print(f'  SLP: {slp_count}  Wind: {wind_count}  Temp: {temp_count}')

# ── Summary table ─────────────────────────────────────────────────────────────
import pandas as pd

_df = pd.DataFrame([{
    'ICAO':       d['icao'],
    'Src':        d.get('source', 'metar'),
    'Name':       d['name'],
    'Time':       d['timestamp'],
    'Lat':        d['lat'],
    'Lon':        d['lon'],
    'Temp(C)':    d['temp'],
    'Dew(C)':     d['dew'],
    'RH(%)':      d['rh'],
    'Wind Dir':   d['wind_dir'],
    'Wind Spd':   d['wind_spd'],
    'Wind Gust':  d['wind_gust'],
    'Vis(SM)':    d['vis'],
    'Wx':         d['weather'],
    'Oktas':      d['oktas'],
    'Ceiling':    d['ceiling'],
    'SLP(hPa)':   d['slp'],
    'SLP Lbl':    d['slp_label'],
    'Tendency':   d.get('tendency'),
    'P Change':   d.get('pressure_change'),
    'Sky Obs':    d['has_sky_obs'],
    'Lowest Sig': d['lowest_sig']['raw'] if d['lowest_sig'] else None,
    'Clouds':     ' '.join(c['raw'] for c in d['clouds']),
    'Cat':        d['flt_cat'],
} for d in metar_records])

def _style_df(df, caption):
    grad_cols = [c for c in ['Temp(C)','Dew(C)'] if c in df.columns]
    slp_cols  = [c for c in ['SLP(hPa)']          if c in df.columns]
    okta_cols = [c for c in ['Oktas']              if c in df.columns]
    s = df.style.set_caption(caption)
    if grad_cols:  s = s.background_gradient(subset=grad_cols, cmap='RdYlBu_r')
    if slp_cols:   s = s.background_gradient(subset=slp_cols,  cmap='coolwarm')
    if okta_cols:  s = s.background_gradient(subset=okta_cols, cmap='Greys')
    s = s.map(lambda v: (
        'color:red;font-weight:bold' if v == 'LIFR' else
        'color:crimson'              if v == 'IFR'  else
        'color:steelblue'            if v == 'MVFR' else
        'color:green'                if v == 'VFR'  else ''), subset=['Cat'])
    s = s.map(lambda v: 'background:#ddeeff;font-style:italic' if v == 'ecmodel' else '',
                   subset=['Src'])
    s = s.format(na_rep='—', precision=1)
    return s.to_html()

_ROWS = 5
_uid  = 'metartbl'
_short_html = _style_df(_df.head(_ROWS), f'METARs + EC Model — showing {_ROWS} of {len(_df)} records')
_full_html  = _style_df(_df,             f'METARs + EC Model — {len(_df)} records total')

display(HTML(f'''
<div id="{_uid}-short">
  {_short_html}
  <button onclick="document.getElementById('{_uid}-short').style.display='none';
      document.getElementById('{_uid}-full').style.display='block';"
    style="margin-top:6px;padding:4px 14px;font-family:monospace;font-size:11px;
    cursor:pointer;border:1px solid #aaa;border-radius:4px;background:#e8f0fe;color:#1a3a6a;">
    ▼ Show all {len(_df)} rows
  </button>
</div>
<div id="{_uid}-full" style="display:none">
  {_full_html}
  <button onclick="document.getElementById('{_uid}-full').style.display='none';
      document.getElementById('{_uid}-short').style.display='block';"
    style="margin-top:6px;padding:4px 14px;font-family:monospace;font-size:11px;
    cursor:pointer;border:1px solid #aaa;border-radius:4px;background:#e8f0fe;color:#1a3a6a;">
    ▲ Collapse
  </button>
</div>'''))


# ── Cell 5b . Compute pressure tendency from 3-hr SLP history ─
# UNCHANGED — now naturally covers both real METAR and EC model records.

from collections import defaultdict

def classify_tendency(slp_now, slp_3h):
    if slp_now is None or slp_3h is None:
        return None, None
    diff   = slp_now - slp_3h
    change = int(round(diff * 10))
    if abs(diff) < 1.0: return 'steady', change
    return ('rising', change) if diff > 0 else ('falling', change)

def classify_tendency_detailed(slp_series):
    if len(slp_series) < 2: return None, None
    slp_vals = [s for _, s in slp_series if s is not None]
    if len(slp_vals) < 2: return None, None
    first = slp_vals[0]; last = slp_vals[-1]; mid = slp_vals[len(slp_vals)//2]
    diff_total = last - first; diff_first = mid - first; diff_last = last - mid
    STEADY = 1.0
    change = int(round(diff_total * 10))
    def sign(x): return 1 if x > STEADY else (-1 if x < -STEADY else 0)
    s1, s2 = sign(diff_first), sign(diff_last)
    if   s1 ==  1 and s2 ==  1: return 'rising',         change
    elif s1 == -1 and s2 == -1: return 'falling',        change
    elif s1 ==  0 and s2 ==  0: return 'steady',         change
    elif s1 ==  1 and s2 == -1: return 'rising_falling', change
    elif s1 == -1 and s2 ==  1: return 'falling_rising', change
    elif s1 ==  1 and s2 ==  0: return 'rising_steady',  change
    elif s1 == -1 and s2 ==  0: return 'falling_steady', change
    elif s1 ==  0 and s2 ==  1: return 'rising',         change
    elif s1 ==  0 and s2 == -1: return 'falling',        change
    else:                        return 'steady',         change

station_slp_series = defaultdict(list)
for d in metar_records:
    if d['slp'] is not None:
        station_slp_series[d['icao']].append((d['timestamp'], d['slp']))
for icao in station_slp_series:
    station_slp_series[icao].sort(key=lambda x: x[0])

tendency_assigned = 0
for d in metar_records:
    series = [(ts, slp) for ts, slp in station_slp_series[d['icao']]
              if ts <= d['timestamp']]
    if len(series) >= 2:
        tend, change = classify_tendency_detailed(series)
        d['tendency']        = tend
        d['pressure_change'] = change
        tendency_assigned   += 1

print(f'✓ Tendency computed for {tendency_assigned} / {len(metar_records)} records')
no_tend = sum(1 for d in metar_records if d['tendency'] is None)
print(f'  No tendency (insufficient history): {no_tend}')

from collections import Counter
tend_counts = Counter(d['tendency'] for d in metar_records if d['tendency'])
for k, v in sorted(tend_counts.items(), key=lambda x: -x[1]):
    print(f'  {k:<20} {v}')

src_counts = Counter(d.get('source','metar') for d in metar_records)
print(f'\n  Source breakdown in metar_records:')
for src, cnt in src_counts.items():
    print(f'    {src:<10} {cnt} records')

# ── Interactive station badge grid ────────────────────────────
_all_stations_5b  = sorted(set(d['icao'] for d in metar_records))
_no_tend_stations = set(d['icao'] for d in metar_records if d['tendency'] is None)
_good_count_5b    = len(_all_stations_5b) - len(_no_tend_stations)

_latest_5b = {}
for d in metar_records:
    if d['icao'] not in _latest_5b or d['timestamp'] > _latest_5b[d['icao']]['timestamp']:
        _latest_5b[d['icao']] = d

def _station_badge_5b(icao):
    d         = _latest_5b.get(icao, {})
    has_gap   = icao in _no_tend_stations
    bg        = '#fff3b0' if has_gap else '#e6faf0'
    bdr       = '#e6a800' if has_gap else '#1a7a3a'
    clr       = '#7a5000' if has_gap else '#145c2c'
    ts        = d.get('timestamp', '—')
    tend      = d.get('tendency', '—') or '—'
    src       = d.get('source', '—')
    gap_str   = '⚠ no tendency (insufficient history)' if has_gap else '✔ tendency computed'
    detail_id = f'tend-detail-{icao}'
    popup_bdr = '#a85c00' if has_gap else '#1a7a3a'
    popup_gap_clr = '#a85c00' if has_gap else '#1a7a3a'
    all_lines = sorted([r for r in metar_records if r['icao'] == icao],
                       key=lambda r: r['timestamp'])
    metar_rows = ''.join(
        f'<tr style="border-bottom:1px solid #eee;">'
        f'<td style="padding:2px 8px;color:#555;white-space:nowrap;">{r["timestamp"]}</td>'
        f'<td style="padding:2px 8px;font-family:monospace;font-size:10px;color:#1a2030;white-space:nowrap;">'
        f'SLP:{r.get("slp","—")} &nbsp; tend:{r.get("tendency","—")}</td>'
        f'</tr>'
        for r in all_lines
    )
    # Build SLP chart data for this station
    slp_points = [(r["timestamp"], r["slp"]) for r in all_lines if r.get("slp") is not None]
    chart_labels = [p[0] for p in slp_points]
    chart_values = [p[1] for p in slp_points]
    chart_id = f'slp-chart-{icao}'
    chart_labels_js = str(chart_labels).replace("'", '"')
    chart_values_js = str(chart_values)
    return (
        f'<span style="display:inline-block;position:relative;margin:2px;">'
        f'<span onclick="'
        f'var p=document.getElementById(\'{detail_id}\');'
        f'document.querySelectorAll(\'.tend-detail-popup\').forEach(function(x){{if(x.id!==\'{detail_id}\')x.style.display=\'none\';}});'
        f'p.style.display=p.style.display===\'none\'?\'block\':\'none\';" '
        f'style="font-family:monospace;font-size:11px;color:{clr};cursor:pointer;'
        f'background:{bg};border:1px solid {bdr};border-radius:3px;'
        f'padding:1px 6px;display:inline-block;">{icao}</span>'
        f'<div id="{detail_id}" class="tend-detail-popup" '
        f'style="display:none;position:absolute;top:20px;left:0;z-index:9999;'
        f'background:#fff;border:2px solid {popup_bdr};border-radius:8px;padding:12px 16px;'
        f'font-family:monospace;font-size:12px;color:#1a2030;'
        f'box-shadow:0 4px 16px rgba(0,0,0,0.25);min-width:320px;max-width:600px;">'
        f'<b style="font-size:13px;">{icao}</b> '
        f'<span style="color:#888;font-size:10px;">{d.get("name","")}</span> '
        f'<span style="color:#888;font-size:10px;">· {src} · {len(all_lines)} obs</span>'
        f'<hr style="margin:4px 0;border:none;border-top:1px solid #ccc;">'
        f'<span style="color:{popup_gap_clr};font-size:10px;">{gap_str}</span>'
        f' &nbsp; <span style="font-size:10px;">latest: {ts} &nbsp; tend: {tend}</span>'
        f'<hr style="margin:4px 0;border:none;border-top:1px solid #eee;">'
        f'<canvas id="{chart_id}" width="460" height="160" '
        f'style="width:100%;max-width:460px;height:160px;margin:8px 0;display:block;"></canvas>'
        f'<script>'
        f'(function(){{'
        f'  var labels = {chart_labels_js};'
        f'  var values = {chart_values_js};'
        f'  var ctx = document.getElementById("{chart_id}");'
        f'  if (!ctx) return;'
        f'  var mn = Math.min.apply(null,values)-1, mx = Math.max.apply(null,values)+1;'
        f'  new Chart(ctx, {{'
        f'    type:"line",'
        f'    data:{{'
        f'      labels:labels,'
        f'      datasets:[{{'
        f'        label:"SLP (hPa)",'
        f'        data:values,'
        f'        borderColor:"#1a4a8a",'
        f'        backgroundColor:"rgba(26,74,138,0.08)",'
        f'        pointBackgroundColor:"#1a4a8a",'
        f'        pointRadius:4,'
        f'        borderWidth:2,'
        f'        tension:0.3,'
        f'        fill:true'
        f'      }}]'
        f'    }},'
        f'    options:{{'
        f'      responsive:false,'
        f'      plugins:{{legend:{{display:false}},'
        f'        tooltip:{{callbacks:{{label:function(c){{return c.parsed.y.toFixed(1)+" hPa";}}}}}}}},'
        f'      scales:{{'
        f'        x:{{ticks:{{font:{{size:9}},maxRotation:45}}}},'
        f'        y:{{min:mn,max:mx,ticks:{{font:{{size:9}}}},title:{{display:true,text:"hPa",font:{{size:9}}}}}}'
        f'      }}'
        f'    }}'
        f'  }});'
        f'}})()'
        f'</script>'
        f'<div style="max-height:160px;overflow-y:auto;">'
        f'<table style="border-collapse:collapse;width:100%;font-size:10px;">'
        f'<tr style="background:#f0f4f8;">'
        f'<th style="padding:2px 8px;text-align:left;">Time</th>'
        f'<th style="padding:2px 8px;text-align:left;">SLP / Tendency</th></tr>'
        f'{metar_rows}</table></div>'
        f'<button onclick="document.getElementById(\'{detail_id}\').style.display=\'none\';event.stopPropagation();" '
        f'style="margin-top:8px;font-size:10px;padding:2px 10px;cursor:pointer;'
        f'border:1px solid #aaa;border-radius:3px;background:#f0f0f0;">✕ close</button>'
        f'</div></span>'
    )

_badge_rows_5b = ''.join(_station_badge_5b(icao) for icao in _all_stations_5b)
_all_ts_5b     = sorted(set(d['timestamp'] for d in metar_records))
_slp_5b        = sum(1 for d in metar_records if d['slp'])
_wind_5b       = sum(1 for d in metar_records if d['wind_dir'] is not None)
_temp_5b       = sum(1 for d in metar_records if d['temp'] is not None)

display(HTML(f'''
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<div style="background:#b6f5c8;border:4px solid #1a7a3a;border-radius:10px;padding:24px 28px;margin:16px 0;overflow:visible;">
  <div style="color:#145c2c;font-size:24px;font-family:monospace;margin-bottom:8px;">
    ✔ {len(_all_stations_5b)} STATIONS WITH DATA &nbsp;|&nbsp; {_good_count_5b} COMPLETE TENDENCY
  </div>
  <div style="color:#1a7a3a;font-size:13px;font-family:monospace;margin-bottom:10px;">
    {len(_all_ts_5b)} timesteps &nbsp;|&nbsp; {len(metar_records)} total records &nbsp;|&nbsp;
    SLP: {_slp_5b} &nbsp; Wind: {_wind_5b} &nbsp; Temp: {_temp_5b}
  </div>
  <div style="line-height:2.2;overflow:visible;">{_badge_rows_5b}</div>
</div>'''))

# ── Cell 5c . Fetch Fort Vermillion (71024) from ogimet ───────────────────────
import requests, re
from datetime import datetime, timezone as _tz

OGIMET_SYNOP_URL = 'https://www.ogimet.com/cgi-bin/getsynop'
FV_WMO   = '71024'
FV_ICAO  = 'CXFV'   # synthetic key — not a real ICAO, but unique in STATIONS
FV_LAT   = 58.3822
FV_LON   = -116.0402 #offset for better viewing. true lon -116.0402
FV_NAME  = 'Fort Vermillion, Alta'

# Register into STATIONS so downstream cells see it
STATIONS[FV_ICAO] = {
    'icao':   FV_ICAO,
    'name':   FV_NAME,
    'lat':    FV_LAT,
    'lon':    FV_LON,
    'tier':   0,
    'source': 'synop',
}

def fetch_ogimet_synop(wmo_id, ndays=2):
    now = datetime.now(_tz.utc)
    from datetime import timedelta
    start = now - timedelta(days=ndays)
    try:
        r = requests.get('https://www.ogimet.com/123456/cgi-bin/getsynop', params={   #123456 is injected to stop the fetching. too pack to show on map
            'block': wmo_id,
            'begin': start.strftime('%Y%m%d%H%M'),
            'end':   now.strftime('%Y%m%d%H%M'),
        }, timeout=20)
        r.raise_for_status()
        return r.text
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        print(f'URL error, by pass this station: {wmo_id}')
        return None

result = fetch_ogimet_synop(FV_WMO)
if result is None:
    print(f'Skipping Fort Vermillion — fetch failed')
else:
    print(result[:1000])


def parse_synop_fm12(line, icao, st):
    m = re.match(r'^\d+,(\d{4}),(\d{2}),(\d{2}),(\d{2}),(\d{2}),(.*)', line)
    if not m:
        return None

    dd, hh = int(m.group(4)), int(m.group(5))  # day, hour — already correct
    # wait — CSV cols are: WMO,YYYY,MM,DD,HH,mm,synop
    yyyy, mo, dd, hh = m.group(1), m.group(2), int(m.group(4)), int(m.group(4))
    # re-extract cleanly
    parts_csv = line.split(',', 6)
    if len(parts_csv) < 7:
        return None
    dd   = int(parts_csv[3])
    hh   = int(parts_csv[4])
    synop_str = parts_csv[6].strip()

    timestamp = f'{dd:02d}{hh:02d}00Z'
    _metar_ts_set = set(d['timestamp'] for d in metar_records)
    if timestamp not in _metar_ts_set:
        return None

    groups = synop_str.replace('=', ' ').split()
    # Find WMO index position and start after it
    try:
        data_start = next(i for i, g in enumerate(groups) if g == '71024') + 1
    except StopIteration:
        data_start = 3
    groups = groups[data_start:]

    temp = dew = slp = wind_dir = wind_spd = None

    for g in groups:
        # Wind group: Nddff or /ddff — N or / then 4 digits
        if re.match(r'^[\d/]\d{4}$', g) and wind_dir is None:
            try:
                dd_ = int(g[1:3]) * 10   # tens of degrees → degrees
                ff  = int(g[3:5])
                if 0 < dd_ <= 360:
                    wind_dir = dd_
                    wind_spd = ff
            except: pass

        # 1sTTT — air temperature  (s=0 positive, s=1 negative)
        elif re.match(r'^1[01]\d{3}$', g):
            try:
                sign = -1 if g[1] == '1' else 1
                temp = sign * int(g[2:]) / 10
            except: pass

        # 2sTTT — dew point  (s=0 positive, s=1 negative)
        elif re.match(r'^2[01]\d{3}$', g):
            try:
                sign = -1 if g[1] == '1' else 1
                dew = sign * int(g[2:]) / 10
            except: pass

        # 3PPPP — station pressure (skip)
        elif re.match(r'^3\d{4}$', g):
            pass

        # 4PPPP — sea level pressure
        elif re.match(r'^4\d{4}$', g):
            try:
                raw = int(g[1:])
                slp = (900 + raw/10) if raw >= 5000 else (1000 + raw/10)
            except: pass

    rh = None
    if temp is not None and dew is not None:
        import math as _m
        a, b = 17.625, 243.04
        rh = round(100 * _m.exp((a*dew/(b+dew)) - (a*temp/(b+temp))))
        rh = max(0, min(100, rh))

    slp_label = f'{int(round(slp*10))%1000:03d}' if slp else ''

    return dict(
        icao=icao, name=st['name'],
        lat=st['lat'], lon=st['lon'],
        source='synop',
        timestamp=timestamp,
        wind_dir=wind_dir, wind_spd=wind_spd, wind_gust=0,
        vis=10.0, temp=temp, dew=dew, rh=rh,
        slp=slp, slp_label=slp_label,
        has_sky_obs=False, oktas=0,
        clouds=[], lowest_sig=None, ceiling=99999,
        weather='', flt_cat='VFR',
        tendency=None, pressure_change=None,
        metar_str=line.strip(),
    )

# ── Fetch and parse ───────────────────────────────────────────────────────────
print(f'Fetching Fort Vermillion (WMO {FV_WMO}) from ogimet...')
fv_records = []
try:
    raw = fetch_ogimet_synop(FV_WMO, ndays=2)
    if raw is None:
        print(f'Skipping Fort Vermillion — fetch returned no data')
    else:
        for line in raw.splitlines():
            if 'AAXX' not in line:
                continue
            rec = parse_synop_fm12(line, FV_ICAO, STATIONS[FV_ICAO])
            if rec:
                fv_records.append(rec)

        # Deduplicate by timestamp — keep latest
        seen = {}
        for r in fv_records:
            seen[r['timestamp']] = r
        fv_records = list(seen.values())

        metar_records.extend(fv_records)
        print(f'✓ Fort Vermillion: {len(fv_records)} obs added → timestamps: {[r["timestamp"] for r in fv_records]}')

except Exception as e:
    print(f'✗ Fort Vermillion fetch failed: {e}')

import pandas as pd

_fv_df = pd.DataFrame([{
    'Timestamp':  r['timestamp'],
    'Temp(C)':    r['temp'],
    'Dew(C)':     r['dew'],
    'RH(%)':      r['rh'],
    'SLP(hPa)':   r['slp'],
    'Wind Dir':   r['wind_dir'],
    'Wind Spd':   r['wind_spd'],
    'Flt Cat':    r['flt_cat'],
    'Source':     r['source'],
} for r in fv_records])

if fv_records:
    _fv_styler = _fv_df.style.set_caption(f'Fort Vermillion (71024 / {FV_ICAO}) — {len(fv_records)} obs')
    if _fv_df['Temp(C)'].notna().any():
        _fv_styler = _fv_styler.background_gradient(subset=['Temp(C)','Dew(C)'], cmap='RdYlBu_r')
    if _fv_df['SLP(hPa)'].notna().any():
        _fv_styler = _fv_styler.background_gradient(subset=['SLP(hPa)'], cmap='coolwarm')
    display(HTML(_fv_styler.format(na_rep='—', precision=1).to_html()))
else:
    display(HTML('<div style="font-family:monospace;color:#888;">Fort Vermillion — no data available</div>'))

# ── Cell 6 . Kriging / RBF interpolation ──────────────────────
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter

def build_grid(records, field, method='rbf', N=220, pad=1.5,
               rbf_smoothing=0.3, sigma=3.0):
    '''
    Interpolate scattered obs onto a regular grid.
    method: 'rbf' uses thin-plate spline (smooth, fast)
            'kriging' uses Ordinary Kriging (best for SLP)
    Returns (grid_2d, lon_vec, lat_vec, lons_flat, lats_flat)
    '''
    pts = [(d['lat'], d['lon'], d[field])
           for d in records if d.get(field) is not None]
    if len(pts) < 8:
        return None, None, None, None, None

    # deduplicate by rounding to 2 decimal degrees (~1km), average values
    _seen = {}
    for la, lo, v in pts:
        key = (round(la, 2), round(lo, 2))
        if key not in _seen:
            _seen[key] = []
        _seen[key].append(v)
    pts = [(k[0], k[1], float(np.mean(vs))) for k, vs in _seen.items()]
    if len(pts) < 8:
        return None, None, None, None, None

    lats = np.array([p[0] for p in pts])
    lons = np.array([p[1] for p in pts])
    vals = np.array([p[2] for p in pts], dtype=float)

    lat_min, lat_max = lats.min()-pad, lats.max()+pad
    lon_min, lon_max = lons.min()-pad, lons.max()+pad

    lon_vec = np.linspace(lon_min, lon_max, N)
    lat_vec = np.linspace(lat_min, lat_max, N)
    glon, glat = np.meshgrid(lon_vec, lat_vec)  # (N,N)

    if method == 'kriging':
        # Ordinary Kriging with linear variogram
        ok = OrdinaryKriging(
            lons, lats, vals,
            variogram_model='linear',
            verbose=False, enable_plotting=False
        )
        z, _ = ok.execute('grid', lon_vec, lat_vec)
        grid = np.array(z)  # (N_lat, N_lon)
    else:
        # RBF thin-plate spline
        obs_xy = np.column_stack([lons, lats])
        try:
            rbf = RBFInterpolator(
                obs_xy, vals,
                kernel='thin_plate_spline',
                smoothing=max(rbf_smoothing * len(pts), 1e-6)
            )
        except np.linalg.LinAlgError:
            rbf = RBFInterpolator(
                obs_xy, vals,
                kernel='linear',
                smoothing=max(rbf_smoothing * len(pts), 1.0)
            )
        qi = np.column_stack([glon.ravel(), glat.ravel()])
        grid = rbf(qi).reshape(N, N)

    # Gaussian smoothing
    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma)

    return grid, lon_vec, lat_vec, lons, lats

print(f'Building SLP grid ({INTERP_METHOD})...')
slp_grid, lon_vec, lat_vec, obs_lons, obs_lats = build_grid(
    metar_records, 'slp',
    method=INTERP_METHOD, N=GRID_N,
    rbf_smoothing=RBF_SMOOTHING, sigma=SIGMA_SMOOTH
)
if slp_grid is not None:
    print(f'✓ SLP grid: {slp_grid.shape}  '
          f'range {slp_grid.min():.1f}-{slp_grid.max():.1f} hPa')
else:
    print('⚠ Not enough SLP data')

print(f'Building temperature grid...')
tmp_grid, tlon_vec, tlat_vec, _, _ = build_grid(
    metar_records, 'temp',
    method='rbf', N=GRID_N,
    rbf_smoothing=0.2, sigma=2.5
)
if tmp_grid is not None:
    print(f'✓ Temp grid: range {tmp_grid.min():.1f}-{tmp_grid.max():.1f} °C')


# ── Cell 7 . Locate H / L pressure centres ────────────────────
import math
from scipy.ndimage import maximum_filter, minimum_filter, label
def find_hl_centers(grid, lon_vec, lat_vec,
                    neighborhood=20, min_delta=2.0,
                    records=None):
    '''
    Find local maxima (H) and minima (L) in grid.
    neighborhood: search radius in grid cells
    min_delta: minimum difference from background to qualify
    Returns list of dicts: {type, lat, lon, val}
    '''
    # extra smooth for extrema detection only
    if records is None:
        records = metar_records
    # extra smooth for extrema detection only
    sg = gaussian_filter(grid, sigma=HL_SIGMA)

    max_f = maximum_filter(sg, size=neighborhood)
    min_f = minimum_filter(sg, size=neighborhood)

    is_max = (sg == max_f) & (sg - min_f > min_delta)
    is_min = (sg == min_f) & (max_f - sg > min_delta)

    centers = []
    for typ, mask in [('H', is_max), ('L', is_min)]:
        lbl, n = label(mask)
        for i in range(1, n+1):
            rows, cols = np.where(lbl==i)
            # pick the actual extremum cell within each labelled blob
            if typ=='H':
                best = np.argmax(sg[rows, cols])
            else:
                best = np.argmin(sg[rows, cols])
            r, c = rows[best], cols[best]
            # bounds check
            if r < neighborhood or r > len(lat_vec)-neighborhood: continue
            if c < neighborhood or c > len(lon_vec)-neighborhood: continue
            # find stations whose interpolated grid value is inside the centre
            # i.e. for H: station grid value >= grid_val - threshold
            #      for L: station grid value <= grid_val + threshold
            _grid_val   = float(grid[r, c])
            _centre_lat = lat_vec[r]
            _centre_lon = lon_vec[c]

            # for each station, look up its value on the interpolated grid
            def _grid_at(sta_lat, sta_lon):
                _ri = int(round((sta_lat - lat_vec[0]) / (lat_vec[-1] - lat_vec[0]) * (len(lat_vec) - 1)))
                _ci = int(round((sta_lon - lon_vec[0]) / (lon_vec[-1] - lon_vec[0]) * (len(lon_vec) - 1)))
                _ri = max(0, min(len(lat_vec) - 1, _ri))
                _ci = max(0, min(len(lon_vec) - 1, _ci))
                return float(grid[_ri, _ci])

            # threshold: one SLP_INTERVAL step inside the centre
            _thresh = SLP_INTERVAL

            if typ == 'H':
                # REPLACE
                _inside = [
                    d['slp'] for d in records
                    if d['slp'] is not None
                    and _grid_at(d['lat'], d['lon']) >= _grid_val - _thresh
                ]
                # REPLACE
                if _inside:
                    _all_nearby = [
                        d['slp'] for d in records
                        if d['slp'] is not None
                        and abs(d['lat'] - _centre_lat) < 5
                        and abs(d['lon'] - _centre_lon) < 8
                    ]
                    _val = round(min(_all_nearby)) if _all_nearby else math.ceil(_grid_val) - 1
                else:
                    _val = math.floor(_grid_val) + 1
            else:
                # REPLACE
                _inside = [
                    d['slp'] for d in records
                    if d['slp'] is not None
                    and _grid_at(d['lat'], d['lon']) <= _grid_val + _thresh
                ]
                if _inside:
                    _all_nearby = [
                        d['slp'] for d in metar_records
                        if d['slp'] is not None
                        and abs(d['lat'] - _centre_lat) < 5
                        and abs(d['lon'] - _centre_lon) < 8
                    ]
                    _val = round(min(_all_nearby)) if _all_nearby else math.ceil(_grid_val) - 1
                else:
                    _val = math.ceil(_grid_val) - 1
            centers.append(dict(
                type=typ,
                lat=lat_vec[r], lon=lon_vec[c],
                # val=float(_val)  hide the H/L value
            ))
    return centers

if slp_grid is not None:
    hl_centers = find_hl_centers(slp_grid, lon_vec, lat_vec,
                                 neighborhood=HL_NEIGHBORHOOD,
                                 min_delta=HL_MIN_DELTA)
    highs = [c for c in hl_centers if c['type']=='H']
    lows  = [c for c in hl_centers if c['type']=='L']
    print(f'✓ Found {len(highs)} High(s) and {len(lows)} Low(s)')
    for c in hl_centers:
        print(f"  {c['type']}  @ {c['lat']:.2f}°N  {c['lon']:.2f}°E")
else:
    hl_centers = []

# ── Cell 7.5 . Per-timestamp SLP grids and H/L centres for JS dropdown ───
import json as _json_ts

_ts_all_pre = sorted(set(d['timestamp'] for d in metar_records if d['timestamp']))
_ts_slp = {}

for _ts in _ts_all_pre:
    _recs = [d for d in metar_records if d['timestamp'] == _ts]

    # build SLP grid for this timestamp
    _grid, _lv, _ltv, _, _ = build_grid(
        _recs, 'slp',
        method=INTERP_METHOD, N=GRID_N,
        rbf_smoothing=RBF_SMOOTHING, sigma=SIGMA_SMOOTH
    )

    if _grid is None:
        _ts_slp[_ts] = {'contours': [], 'hl': []}
        print(f'  {_ts}: insufficient SLP data, skipped')
        continue

    # contours
    _glon, _glat = np.meshgrid(_lv, _ltv)
    _slp_min = np.floor(_grid.min() / SLP_INTERVAL) * SLP_INTERVAL
    _slp_max = np.ceil(_grid.max()  / SLP_INTERVAL) * SLP_INTERVAL
    _levels  = np.arange(_slp_min, _slp_max + SLP_INTERVAL, SLP_INTERVAL)
    _fig, _ax = plt.subplots(figsize=(1, 1))
    _cs = _ax.contour(_glon, _glat, _grid, levels=_levels)
    plt.close(_fig)

    _contours = []
    for _li, _lvl in enumerate(_cs.levels):
        _is_major = (int(_lvl) % 20 == 0)
        _weight   = 2.5 if _is_major else (1.4 if int(_lvl) % 8 == 0 else 0.7)
        _opacity  = 0.95 if _is_major else (0.65 if int(_lvl) % 8 == 0 else 0.40)
        for _coords in _cs.allsegs[_li]:
            if len(_coords) < 2: continue
            # label position: midpoint of segment
            _mid = _coords[len(_coords) // 2]
            _contours.append({
                'level':   float(_lvl),
                'weight':  _weight,
                'opacity': _opacity,
                'coords':  [[float(c[0]), float(c[1])] for c in _coords],
                'label_lon': float(_mid[0]),
                'label_lat': float(_mid[1]),
            })

    # H/L centres
    # REPLACE
    _hl = find_hl_centers(_grid, _lv, _ltv,
                          neighborhood=HL_NEIGHBORHOOD,
                          min_delta=HL_MIN_DELTA,
                          records=_recs)

    _ts_slp[_ts] = {'contours': _contours, 'hl': _hl}
    print(f'  {_ts}: {len(_contours)} contour segments, '
          f'{sum(1 for x in _hl if x["type"]=="H")} H, '
          f'{sum(1 for x in _hl if x["type"]=="L")} L')

_ts_slp_json_str = _json_ts.dumps(_ts_slp)
print(f'✓ Per-timestamp SLP/HL ready for {len(_ts_slp)} timestamps')

#1800z auto switch
# 1. No countour
# -- Cell 9 - Interactive Folium map with OSM tiles ---
import folium
from folium import Element
import json as _json
import numpy as np
from matplotlib import pyplot as plt
import math as _math

# -- build the map ---
center_lat = 56
center_lon = -114
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles=None,
    prefer_canvas=True
)

# tile layers — Blank added last so Leaflet selects it as default
folium.TileLayer(
    tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    attr='CartoDB Positron', name='White (CartoDB)', max_zoom=19
).add_to(m)
folium.TileLayer(tiles='OpenStreetMap', name='OpenStreetMap', max_zoom=19).add_to(m)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    attr='ESRI World Topo', name='ESRI Topo', max_zoom=19
).add_to(m)
folium.TileLayer(
    tiles='about:blank',
    attr='Blank', name='Blank (borders only)', max_zoom=19
).add_to(m)
# white background + force blank as default on load
m.get_root().html.add_child(Element(
    '<style>.leaflet-container{background:#ffffff!important;}</style>\n'
    '<script>\n'
    '(function(){\n'
    '  function initBlank(){\n'
    '    var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    if(!keys.length){setTimeout(initBlank,200);return;}\n'
    '    var MAP=window[keys[0]];\n'
    '    var blankLayer=null;\n'
    '    var others=[];\n'
    '    MAP.eachLayer(function(l){\n'
    '      if(l instanceof L.TileLayer){\n'
    '        if(l.options.name==="Blank (borders only)" || (l._url&&l._url==="about:blank")){\n'
    '          blankLayer=l;\n'
    '        } else {\n'
    '          others.push(l);\n'
    '        }\n'
    '      }\n'
    '    });\n'
    '    others.forEach(function(l){MAP.removeLayer(l);});\n'
    '    if(blankLayer) blankLayer.addTo(MAP);\n'
    '  }\n'
    '  function tryInitBlank(){\n'
    '    var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    if(!keys.length){setTimeout(tryInitBlank,200);return;}\n'
    '    var MAP=window[keys[0]];\n'
    '    // also fire on every layeradd to catch late-loading tiles\n'
    '    MAP.on("layeradd", function(){\n'
    '      MAP.eachLayer(function(l){\n'
    '        if(l instanceof L.TileLayer && l._url && l._url!=="about:blank"){\n'
    '          MAP.removeLayer(l);\n'
    '        }\n'
    '      });\n'
    '    });\n'
    '    initBlank();\n'
    '  }\n'
    '  if(document.readyState==="complete"){setTimeout(tryInitBlank,100);}\n'
    '  else{window.addEventListener("load",function(){setTimeout(tryInitBlank,100);});}\n'
    '})();\n'
    '</script>\n'
))

# ---- SLP CONTOURS --------------------------------------------------------
slp_fg = folium.FeatureGroup(name='SLP Isobars', show=None)

if slp_grid is not None:
    glon, glat = np.meshgrid(lon_vec, lat_vec)
    slp_min = np.floor(slp_grid.min() / SLP_INTERVAL) * SLP_INTERVAL
    slp_max = np.ceil(slp_grid.max()  / SLP_INTERVAL) * SLP_INTERVAL
    levels  = np.arange(slp_min, slp_max + SLP_INTERVAL, SLP_INTERVAL)
    fig_c, ax_c = plt.subplots(figsize=(1, 1))
    cs = ax_c.contour(glon, glat, slp_grid, levels=levels)
    plt.close(fig_c)
    for lvl_idx, lvl in enumerate(cs.levels):
        is_major = (int(lvl) % 20 == 0)
        weight   = 2.5 if is_major else (1.4 if int(lvl)%8==0 else 0.7)
        opacity  = 0.95 if is_major else (0.65 if int(lvl)%8==0 else 0.40)
        for coords in cs.allsegs[lvl_idx]:
            if len(coords) < 2:
                continue
            geo_coords = [[float(c[0]), float(c[1])] for c in coords]
            feature = {
                'type': 'Feature',
                'geometry': {'type': 'LineString', 'coordinates': geo_coords},
                'properties': {'level': float(lvl)}
            }
            folium.GeoJson(
                feature,
                style_function=lambda f: {
                'color': '#000000', 'weight': 1, 'opacity': 1.0
                },
                tooltip=folium.Tooltip(f'{int(lvl)} ')
            ).add_to(slp_fg)
            if int(lvl) % 4 == 0 and len(coords) > 4:
                mid = coords[len(coords)//2]
                folium.Marker(
                    location=[float(mid[1]), float(mid[0])],
                    icon=folium.DivIcon(
                        html=(f'<div style="font-size:9px;font-weight:900;color:#1a3a6a;'
                              f'font-family:Courier New,monospace;white-space:nowrap;'
                              f'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,'
                              f'1px -1px 0 #fff,-1px 1px 0 #fff;">'
                              f'{int(lvl)}</div>'),
                        icon_size=(32, 14), icon_anchor=(16, 7)
                    )
                ).add_to(slp_fg)

# ---- H/L MARKERS ---------------------------------------------------------
hl_fg = folium.FeatureGroup(name='H/L Centers', show=None)
if hl_centers:
    for c in hl_centers:
        color  = 'black'
        shadow = '1px 1px 0 white,-1px -1px 0 white,1px -1px 0 white,-1px 1px 0 white'
        html = (
            f'<div style="display:flex;flex-direction:column;align-items:center;pointer-events:none">'
            f'<div style="font-size:59px;font-weight:900;color:{color};'
            f'font-family:Palatino Linotype,Palatino,serif;line-height:1;text-shadow:{shadow};">{c["type"]}</div>'
            f'</div>'
        )
        folium.Marker(
            location=[c['lat'], c['lon']],
            icon=folium.DivIcon(html=html, icon_size=(60, 44), icon_anchor=(30, 12)),
            tooltip=c['type']
        ).add_to(hl_fg)


# ---- STATION MODELS ------------------------------------------------------
stn_fg = folium.FeatureGroup(name='Station Models', show=True)
visible = list(metar_records)

for d in visible:
    svg_str, sw, sh = station_model_svg(d, S=SYMBOL_SCALE)
    fc_color = flight_cat_color(d)
    popup_html = (
        f'<div style="font-family:monospace;font-size:12px;min-width:200px">'
        f'<b style="font-size:14px;color:#1a4a8a">{d["icao"]}</b> '
        f'<span style="color:{fc_color};font-weight:bold">{d["flt_cat"]}</span><br>'
        f'<span style="color:#888;font-size:10px">{d["name"]}</span><br>'
        f'<hr style="margin:4px 0">'
        f'Temp: <b>{d["temp"]}C</b> &nbsp; Dew: <b>{d["dew"]}C</b><br>'
        f'Wind: <b>{d["wind_dir"]}/{d["wind_spd"]}kt'
        + (f' G{d["wind_gust"]}' if d.get('wind_gust') else '')
        + f'</b><br>'
        f'Vis: <b>{d["vis"]} SM</b> &nbsp; Wx: <b>{d["weather"] or "NIL"}</b><br>'
        f'SLP: <b>{d["slp"]} hPa</b> &nbsp; RH: <b>{d["rh"]}%</b><br>'
        f'Cloud: <b>' + ' '.join(c['raw'] for c in d['clouds']) + '</b><br>'
        f'<a href="https://aviationweather.gov/api/data/metar?ids={d["icao"]}&hours=24&taf=1" '
        f'target="_blank" style="font-size:10px;color:#1a4a8a;">METAR+TAF: {d["icao"]} ↗</a></div>'
    )
    folium.Marker(
        location=[d['lat'], d['lon']],
        icon=folium.DivIcon(html=svg_str, icon_size=(sw, sh),
                            icon_anchor=(sw/2, sh/2), class_name=''),
        popup=folium.Popup(popup_html, max_width=260),
        tooltip=f'{d["icao"]} {d["temp"]}C/{d["dew"]}C SLP={d["slp"]}'
    ).add_to(stn_fg)
# stn_fg.add_to(m)  # disabled: JS dropdown controls station rendering

# ---- BUILD PER-TIMESTAMP STATION DATA for JS dropdown --------------------
import json as _json2
_ts_all = sorted(set(d['timestamp'] for d in metar_records if d['timestamp']))
_ts_data = {}
for _ts in _ts_all:
    _entries = []
    for _d in metar_records:
        if _d['timestamp'] != _ts: continue
        pass  # no geographic filter — use all stations
        _svg, _sw, _sh = station_model_svg(_d, S=SYMBOL_SCALE)
        _fc = flight_cat_color(_d)
        _wg = f' G{_d["wind_gust"]}' if _d.get('wind_gust') else ''
        _pop = (
            f'<div style="font-family:monospace;font-size:12px;min-width:200px">'
            f'<b style="font-size:14px;color:#1a4a8a">{_d["icao"]}</b> '
            f'<span style="color:{_fc};font-weight:bold">{_d["flt_cat"]}</span><br>'
            f'<span style="color:#888;font-size:10px">{_d["name"]}</span>'
            f'<hr style="margin:4px 0">'
            f'Temp/Dew: <b>{_d["temp"]}C / {_d["dew"]}C</b><br>'
            f'Wind: <b>{_d["wind_dir"]}/{_d["wind_spd"]}kt{_wg}</b><br>'
            f'Vis: <b>{_d["vis"]} SM</b> Wx: <b>{_d["weather"] or "NIL"}</b><br>'
            f'SLP: <b>{_d["slp"]} hPa</b> RH: <b>{_d["rh"]}%</b><br>'
            f'Cloud: <b>' + ' '.join(c['raw'] for c in _d['clouds']) + '</b><br>'
            f'<a href="https://aviationweather.gov/api/data/metar?ids={_d["icao"]}&hours=24&taf=1" '
            f'target="_blank" style="font-size:10px;color:#1a4a8a;">METAR+TAF: {_d["icao"]} ↗</a></div>'
        )
        _entries.append({
            'lat': _d['lat'], 'lon': _d['lon'],
            'svg': _svg, 'sw': _sw, 'sh': _sh, 'popup': _pop,
            'tip': f'{_d["icao"]} {_d["temp"]}C/{_d["dew"]}C {_d["wind_dir"]}/{_d["wind_spd"]}kt'
        })
    _ts_data[_ts] = _entries
_ts_json_str = _json2.dumps(_ts_data)
_ts_list_str = _json2.dumps(_ts_all)
_latest_ts   = _ts_all[-1] if _ts_all else ''
print(f'Timestamps available: {_ts_all}')
# ---- end per-timestamp data -----------------------------------------------

folium.LayerControl(collapsed=False).add_to(m)



# ---- TIMESTEP DROPDOWN BAR -----------------------------------------------
ts_bar_html = (
    '<div id="syn-ts-bar" style="'
    'position:fixed;bottom:26px;left:10px;z-index:10000;'
    'background:rgba(255,255,255,0.96);border:1px solid #ccc;border-radius:8px;'
    'padding:6px 12px;font-family:Courier New,monospace;font-size:12px;'
    'box-shadow:0 2px 10px rgba(0,0,0,0.15);display:flex;align-items:center;gap:8px;">'
    '<b style="color:#1a4a8a">Time:</b>'
    '<select id="ts-select" onchange="synUpdateTS(this.value)" '
    'style="font-family:Courier New,monospace;font-size:12px;padding:2px 5px;'
    'border:1px solid #aac;border-radius:4px;background:#f8fbff;color:#1a4a8a;cursor:pointer">'
    '</select>'
    '<span id="ts-count" style="color:#888;font-size:10px;min-width:60px"></span>'
    '<button id="btn-slp" onclick="synToggleLayer(\'slp\')" '
    'style="font-size:9px;padding:2px 7px;cursor:pointer;border:1px solid #aaa;'
    'border-radius:3px;background:#e8f0fe;color:#1a3a6a">Isobars ✓</button>'
    '<button id="btn-hl" onclick="synToggleLayer(\'hl\')" '
    'style="font-size:9px;padding:2px 7px;cursor:pointer;border:1px solid #aaa;'
    'border-radius:3px;background:#e8f0fe;color:#1a3a6a">H/L ✓</button>'
    '<button id="btn-svg" onclick="synToggleLayer(\'svg\')" '
    'style="font-size:9px;padding:2px 7px;cursor:pointer;border:1px solid #aaa;'
    'border-radius:3px;background:#e8f0fe;color:#1a3a6a">Stn ✓</button>'
    '</div>'
)
m.get_root().html.add_child(Element(ts_bar_html))

# ---- TIMESTEP JS --------------------------------------------------------
ts_js = (
    '<script>\n'
    'var _SYN_TS_DATA = ' + _ts_json_str + ';\n'
    'var _SYN_TS_LIST = ' + _ts_list_str + ';\n'
    'var _SYN_SLP     = ' + _ts_slp_json_str + ';\n'
    'var _synStnLayer = null;\n'
    'var _synSlpLayer = null;\n'
    'var _synHLLayer  = null;\n'
    'function synUpdateTS(ts) {\n'
    '  var entries = _SYN_TS_DATA[ts] || [];\n'
    '  var countEl = document.getElementById("ts-count");\n'
    '  if (countEl) countEl.textContent = entries.length + " stns";\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) { console.warn("synUpdateTS: map not ready"); return; }\n'
    '  var MAP = window[keys[0]];\n'
    '  if (!MAP || typeof MAP.removeLayer !== "function") { console.warn("synUpdateTS: invalid map"); return; }\n'
'  if (_synStnLayer) { MAP.removeLayer(_synStnLayer); _synStnLayer = null; }\n'
    '  if (_synSlpLayer) { MAP.removeLayer(_synSlpLayer); _synSlpLayer = null; }\n'
    '  if (_synHLLayer)  { MAP.removeLayer(_synHLLayer);  _synHLLayer  = null; }\n'
    '  _synStnLayer = L.layerGroup();\n'
    '  entries.forEach(function(d) {\n'
    '    L.marker([d.lat, d.lon], {\n'
    '      icon: L.divIcon({\n'
    '        html: d.svg, iconSize:[d.sw,d.sh], iconAnchor:[d.sw/2,d.sh/2], className:""\n'
    '      }), zIndexOffset:100\n'
    '    }).bindPopup(d.popup,{maxWidth:280,closeButton:true}).bindTooltip(d.tip).addTo(_synStnLayer);\n'
    '  });\n'
    '  _synStnLayer.addTo(MAP);\n'
    '  var slpData = _SYN_SLP[ts] || {contours:[], hl:[]};\n'
    '  _synSlpLayer = L.layerGroup();\n'
    '  slpData.contours.forEach(function(ct) {\n'
    '    var latlngs = ct.coords.map(function(c){return [c[1],c[0]];});\n'
    '    L.polyline(latlngs, {\n'
    '      color:"#000000", weight:1, opacity:1.0\n'
    '    }).bindTooltip(Math.round(ct.level)+" ").addTo(_synSlpLayer);\n'
'    if (Math.round(ct.level) % 4 === 0) {\n'
    '      L.marker([ct.label_lat, ct.label_lon], {\n'
    '        icon: L.divIcon({\n'
'          html: \'<div style="font-size:13px;font-weight:normal;color:#000000;\'\n'
    '               +\'font-family:Courier New,monospace;white-space:nowrap;\'\n'
    '               +\'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,\'\n'
    '               +\'1px -1px 0 #fff,-1px 1px 0 #fff;">\'\n'
    '               + Math.round(ct.level) + \'</div>\',\n'
    '          iconSize:[42,18], iconAnchor:[21,9], className:""\n'
    '        })\n'
    '      }).addTo(_synSlpLayer);\n'
    '    }\n'
    '  });\n'
    '  var LAT_LINE = 66.0;\n'
    '  slpData.contours.forEach(function(ct) {\n'
    '    var coords = ct.coords;\n'
    '    for (var ii = 0; ii < coords.length - 1; ii++) {\n'
    '      var lat0 = coords[ii][1],   lon0 = coords[ii][0];\n'
    '      var lat1 = coords[ii+1][1], lon1 = coords[ii+1][0];\n'
    '      if ((lat0 <= LAT_LINE && lat1 >= LAT_LINE) ||\n'
    '          (lat0 >= LAT_LINE && lat1 <= LAT_LINE)) {\n'
    '        var t    = (LAT_LINE - lat0) / (lat1 - lat0);\n'
    '        var lonX = lon0 + t * (lon1 - lon0);\n'
    '        L.marker([LAT_LINE, lonX], {\n'
    '          icon: L.divIcon({\n'
    '            html: \'<div style="font-size:13px;font-weight:normal;color:#000000;\'\n'
    '                 +\'font-family:Courier New,monospace;white-space:nowrap;\'\n'
    '                 +\'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,\'\n'
    '                 +\'1px -1px 0 #fff,-1px 1px 0 #fff;">\'\n'
    '                 + Math.round(ct.level) + \'</div>\',\n'
    '            iconSize:[42,18], iconAnchor:[21,9], className:""\n'
    '          })\n'
    '        }).addTo(_synSlpLayer);\n'
    '        break;\n'
    '      }\n'
    '    }\n'
    '  });\n'
    '  var LON_LINE = -125.0;\n'
    '  slpData.contours.forEach(function(ct) {\n'
    '    var coords = ct.coords;\n'
    '    for (var ii = 0; ii < coords.length - 1; ii++) {\n'
    '      var lat0 = coords[ii][1],   lon0 = coords[ii][0];\n'
    '      var lat1 = coords[ii+1][1], lon1 = coords[ii+1][0];\n'
    '      if ((lon0 <= LON_LINE && lon1 >= LON_LINE) ||\n'
    '          (lon0 >= LON_LINE && lon1 <= LON_LINE)) {\n'
    '        var t    = (LON_LINE - lon0) / (lon1 - lon0);\n'
    '        var latX = lat0 + t * (lat1 - lat0);\n'
    '        L.marker([latX, LON_LINE], {\n'
    '          icon: L.divIcon({\n'
    '            html: \'<div style="font-size:13px;font-weight:normal;color:#000000;\'\n'
    '                 +\'font-family:Courier New,monospace;white-space:nowrap;\'\n'
    '                 +\'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,\'\n'
    '                 +\'1px -1px 0 #fff,-1px 1px 0 #fff;">\'\n'
    '                 + Math.round(ct.level) + \'</div>\',\n'
    '            iconSize:[42,18], iconAnchor:[21,9], className:""\n'
    '          })\n'
    '        }).addTo(_synSlpLayer);\n'
    '        break;\n'
    '      }\n'
    '    }\n'
    '  });\n'
    '  _synHLLayer = L.layerGroup();\n'
    '  slpData.hl.forEach(function(c) {\n'
    '    var color = "black";\n'
    '    var shadow = "1px 1px 0 white,-1px -1px 0 white,1px -1px 0 white,-1px 1px 0 white";\n'
    '    var html = \'<div style="display:flex;flex-direction:column;align-items:center;">\'\n'
    '             + \'<div style="font-size:59px;font-weight:900;color:\'+color+\';\'\n'
    '             + \'font-family:Palatino Linotype,Palatino,serif;line-height:1;text-shadow:\'+shadow+\';">\'+c.type+\'</div>\'\n'
    '             + \'</div>\';\n'
    '    L.marker([c.lat, c.lon], {\n'
    '      icon: L.divIcon({html:html, iconSize:[60,44], iconAnchor:[30,12], className:""}),\n'
    '      zIndexOffset: 200\n'
    '    }).bindTooltip(c.type).addTo(_synHLLayer);\n'
    '  });\n'
    '  if (_synShowSlp) _synSlpLayer.addTo(MAP);\n'   # ← ADD THIS
    '  if (_synShowHL)  _synHLLayer.addTo(MAP);\n'    # ← ADD THIS
    '}\n'
    'var _synShowSlp  = true;\n'
    'var _synShowHL   = true;\n'
    'var _synSvgMode  = "colour";\n'
    'function synToggleLayer(which) {\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (which === "slp") {\n'
    '    _synShowSlp = !_synShowSlp;\n'
    '    var btn = document.getElementById("btn-slp");\n'
    '    if (_synShowSlp) {\n'
    '      if (_synSlpLayer) _synSlpLayer.addTo(MAP);\n'
    '      btn.textContent = "Isobars ✓"; btn.style.background = "#e8f0fe";\n'
    '    } else {\n'
    '      if (_synSlpLayer) MAP.removeLayer(_synSlpLayer);\n'
    '      btn.textContent = "Isobars ✗"; btn.style.background = "#f0f0f0";\n'
    '    }\n'
    '  } else if (which === "hl") {\n'
    '    _synShowHL = !_synShowHL;\n'
    '    var btn2 = document.getElementById("btn-hl");\n'
    '    if (_synShowHL) {\n'
    '      if (_synHLLayer) _synHLLayer.addTo(MAP);\n'
    '      btn2.textContent = "H/L ✓"; btn2.style.background = "#e8f0fe";\n'
    '    } else {\n'
    '      if (_synHLLayer) MAP.removeLayer(_synHLLayer);\n'
    '      btn2.textContent = "H/L ✗"; btn2.style.background = "#f0f0f0";\n'
    '    }\n'
    '  } else if (which === "svg") {\n'
    '    _synSvgMode = (_synSvgMode === "colour") ? "wmo" : "colour";\n'
    '    var btn3 = document.getElementById("btn-svg");\n'
    '    var styleEl = document.getElementById("syn-wx-style");\n'
    '    if (!styleEl) {\n'
    '      styleEl = document.createElement("style");\n'
    '      styleEl.id = "syn-wx-style";\n'
    '      document.head.appendChild(styleEl);\n'
    '    }\n'
    '    if (_synSvgMode === "wmo") {\n'
    '      styleEl.textContent = ".syn-wx-box { display: none !important; }";\n'
    '      btn3.textContent = "WMO ✓"; btn3.style.background = "#f4e8c8"; btn3.style.color = "#5c2e00"; btn3.style.borderColor = "#9a6a00";\n'
    '    } else {\n'
    '      styleEl.textContent = "";\n'
    '      btn3.textContent = "Stn ✓"; btn3.style.background = "#e8f0fe"; btn3.style.color = "#1a3a6a"; btn3.style.borderColor = "#aaa";\n'
    '    }\n'
    '  }\n'
    '}\n'
    'function synTsToUtc(ts) {\n'
    '  var clean=ts.replace(/Z$/i,""); var dd=parseInt(clean.slice(0,2),10);\n'
    '  var hh=parseInt(clean.slice(2,4),10); var mn=parseInt(clean.slice(4,6)||"0",10);\n'
    '  var now=new Date();\n'
    '  var d=new Date(Date.UTC(now.getUTCFullYear(),now.getUTCMonth(),dd,hh,mn,0));\n'
    '  if(dd-now.getUTCDate()>15) d.setUTCMonth(d.getUTCMonth()-1);\n'
    '  return d;\n'
    '}\n'
'function synInitDropdown() {\n'
    '  var sel = document.getElementById("ts-select");\n'
    '  if (!sel) { setTimeout(synInitDropdown, 200); return; }\n'
    '  var nowUtc = new Date();\n'
    '  var latest = null;\n'
    '  _SYN_TS_LIST.forEach(function(ts) {\n'
    '    if (synTsToUtc(ts) > nowUtc) return;\n'
    '    var opt = document.createElement("option");\n'
    '    opt.value = ts; opt.textContent = ts;\n'
    '    sel.appendChild(opt);\n'
    '    latest = ts;\n'
    '  });\n'
    '  if (!latest && _SYN_TS_LIST.length) latest = _SYN_TS_LIST[0];\n'
    '  if (latest) { sel.value = latest; synUpdateTS(latest); }\n'
    '}\n'
    'if (document.readyState==="complete") { setTimeout(synInitDropdown,500); }\n'
    'else { window.addEventListener("load",function(){setTimeout(synInitDropdown,500);}); }\n'
    '</script>\n'
)
m.get_root().html.add_child(Element(ts_js))
# ---- end timestep JS ------------------------------------------------------

# ---- JS: load Natural Earth borders --------------------------------------
borders_js = (
    '<style>\n'
    '.leaflet-container { background: #ffffff !important; }\n'
    '</style>\n'
    '<script>\n'
    '(function() {\n'
    '  function loadBorders() {\n'
    '    var items = [\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_coastline.geojson",\n'
    '       {color:"#444",weight:1.8,opacity:1.0,fill:false}],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_boundary_lines_land.geojson",\n'
    '       {color:"#333",weight:2.0,opacity:1.0,fill:false}],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces_lines.geojson",\n'
    '       {color:"#777",weight:0.9,opacity:0.85,fill:false}],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson",\n'
    '       "ALBERTA"],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_lakes.geojson",\n'
    '       {color:"#5588aa",weight:0.8,opacity:0.9,fill:false}]\n'
    '    ];\n'
    '    var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    if (!keys.length) { setTimeout(loadBorders, 200); return; }\n'
    '    var MAP = window[keys[0]];\n'
    '    items.forEach(function(item) {\n'
    '      if (item[1] === "ALBERTA") {\n'
    '        fetch(item[0]).then(function(r){return r.json();}).then(function(gj){\n'
    '          var ab = {type:"FeatureCollection", features: gj.features.filter(function(f){\n'
    '            var n = (f.properties.name || f.properties.NAME || "").toUpperCase();\n'
    '            return n === "ALBERTA";\n'
    '          })};\n'
    '          L.geoJSON(ab,{style:function(){return {color:"#cc0000",weight:3.5,opacity:1.0,fill:false};}}).addTo(MAP);\n'
    '        }).catch(function(e){console.warn("Alberta border load failed",e);});\n'
    '      } else {\n'
    '        fetch(item[0]).then(function(r){return r.json();}).then(function(gj){\n'
    '          L.geoJSON(gj,{style:function(){return item[1];}}).addTo(MAP);\n'
    '        }).catch(function(e){console.warn("border load failed",e);});\n'
    '      }\n'
    '    });\n'
    '  }\n'
    '  if (document.readyState==="complete") { setTimeout(loadBorders,600); }\n'
    '  else { window.addEventListener("load",function(){setTimeout(loadBorders,600);}); }\n'
    '})();\n'
    '</script>'
)
m.get_root().html.add_child(Element(borders_js))

# ---- SAVE PNG BUTTON -----------------------------------------------------
save_btn_html = (
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>\n'
    '<script>\n'
'function synSavePNG() {\n'
    '  var btn    = document.getElementById("btn-save-png");\n'
    '  var status = document.getElementById("save-status");\n'
    '  btn.disabled = true;\n'
    '  btn.textContent = "Capturing...";\n'
    '  status.textContent = "";\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) { status.textContent="Map not found"; btn.disabled=false; return; }\n'
    '  var MAP   = window[keys[0]];\n'
    '  var mapEl = document.getElementById(keys[0]) || document.querySelector(".leaflet-container");\n'
    '  if (!mapEl) { status.textContent="Map el not found"; btn.disabled=false; return; }\n'
    '  var hideEls = [\n'
    '    mapEl.querySelector(".leaflet-control-container"),\n'
    '    document.querySelector(".leaflet-control-layers"),\n'
    '    document.querySelector(".leaflet-control-zoom"),\n'
    '    document.querySelector(".leaflet-control-attribution"),\n'
    '    document.getElementById("syn-ts-bar"),\n'
    '    document.getElementById("syn-save-bar"),\n'
    '    document.getElementById("syn-fs-btn")\n'
    '  ].filter(Boolean);\n'
    '  var prevVis = hideEls.map(function(el){ return el.style.visibility; });\n'
    '  hideEls.forEach(function(el){ el.style.visibility="hidden"; });\n'
    '  html2canvas(mapEl, {\n'
    '    useCORS: true, allowTaint: true, scale: 2, logging: false\n'
    '  }).then(function(canvas) {\n'
    '    hideEls.forEach(function(el,i){ el.style.visibility=prevVis[i]; });\n'
    '    // convert lat/lon crop bounds to pixel coords using Leaflet\n'
    '    var mapRect = mapEl.getBoundingClientRect();\n'
    '    var scale   = 2;\n'
    '    var tl = MAP.latLngToContainerPoint([65.71423563397606, -134.28652654118332]);\n'
    '    var br = MAP.latLngToContainerPoint([44.51251256829053,  -98.50745737363381]);\n'
    '    var x1 = Math.max(0, tl.x) * scale;\n'
    '    var y1 = Math.max(0, tl.y) * scale;\n'
    '    var x2 = Math.min(mapEl.offsetWidth,  br.x) * scale;\n'
    '    var y2 = Math.min(mapEl.offsetHeight, br.y) * scale;\n'
    '    var cropW = x2 - x1;\n'
    '    var cropH = y2 - y1;\n'
    '    if (cropW <= 0 || cropH <= 0) {\n'
    '      status.textContent = "Crop out of view — pan map first";\n'
    '      btn.disabled=false; btn.textContent="\\uD83D\\uDCF7 Save PNG"; return;\n'
    '    }\n'
    '    var out = document.createElement("canvas");\n'
    '    out.width  = cropW;\n'
    '    out.height = cropH;\n'
    '    out.getContext("2d").drawImage(canvas, x1, y1, cropW, cropH, 0, 0, cropW, cropH);\n'
    '    var ts   = (document.getElementById("ts-select")||{}).value||"synoptic";\n'
    '    var name = "synoptic_" + ts.replace(/[^a-zA-Z0-9]/g,"_") + ".png";\n'
    '    var link = document.createElement("a");\n'
    '    link.download = name;\n'
    '    link.href = out.toDataURL("image/png");\n'
    '    link.click();\n'
    '    btn.disabled=false; btn.textContent="\\uD83D\\uDCF7 Save PNG";\n'
    '    status.textContent="Saved!"; setTimeout(function(){ status.textContent=""; },3000);\n'
    '  }).catch(function(e) {\n'
    '    hideEls.forEach(function(el,i){ el.style.visibility=prevVis[i]; });\n'
    '    status.textContent="Failed: "+e.message;\n'
    '    btn.disabled=false; btn.textContent="\\uD83D\\uDCF7 Save PNG";\n'
    '  });\n'
    '}\n'
    '</script>\n'
)
m.get_root().html.add_child(Element(save_btn_html))

# ---- FULLSCREEN BUTTON ---------------------------------------------------
fullscreen_html = (
    '<style>\n'
    '#syn-fs-btn {\n'
    '  position:fixed;top:10px;left:10px;z-index:10001;\n'
    '  background:rgba(255,255,255,0.96);border:1px solid #aaa;border-radius:6px;\n'
    '  padding:5px 10px;font-family:Courier New,monospace;font-size:12px;\n'
    '  box-shadow:0 2px 8px rgba(0,0,0,0.15);cursor:pointer;color:#1a3a6a;\n'
    '}\n'
    '#syn-fs-btn:hover { background:#e8f0fe; }\n'
    '.syn-fs-active {\n'
    '  position:fixed!important;top:0!important;left:0!important;\n'
    '  width:100vw!important;height:100vh!important;\n'
    '  z-index:9999!important;margin:0!important;\n'
    '}\n'
    '</style>\n'
    '<button id="syn-fs-btn" onclick="synToggleFS()">&#x26F6; Fullscreen</button>\n'
    '<script>\n'
    'var _synFS = false;\n'
    'var _synMapEl = null;\n'
    'var _synOrigStyle = "";\n'
    'function synToggleFS() {\n'
    '  var btn = document.getElementById("syn-fs-btn");\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) { console.warn("map not found"); return; }\n'
    '  var MAP = window[keys[0]];\n'
    '  if (!_synMapEl) {\n'
    '    _synMapEl = document.getElementById(keys[0]);\n'
    '    if (!_synMapEl) _synMapEl = document.querySelector(".leaflet-container");\n'
    '  }\n'
    '  if (!_synMapEl) { console.warn("map element not found"); return; }\n'
    '  _synFS = !_synFS;\n'
    '  if (_synFS) {\n'
    '    _synOrigStyle = _synMapEl.getAttribute("style") || "";\n'
    '    _synMapEl.setAttribute("style",\n'
    '      "position:fixed!important;top:0;left:0;"\n'
    '      +"width:100vw!important;height:100vh!important;"\n'
    '      +"z-index:9999!important;margin:0!important;");\n'
    '    btn.innerHTML = "&#x274C; Exit Fullscreen";\n'
    '  } else {\n'
    '    _synMapEl.setAttribute("style", _synOrigStyle);\n'
    '    btn.innerHTML = "&#x26F6; Fullscreen";\n'
    '  }\n'
    '  setTimeout(function(){ MAP.invalidateSize(); }, 100);\n'
    '}\n'
    '</script>\n'
)
m.get_root().html.add_child(Element(fullscreen_html))
m.get_root().html.add_child(Element(fire_zones_html))

# save & display
out_path = 'output/synoptic_map.html'
m.save(out_path)
print(f'Map saved: {out_path}')
print(f'Stations: {len(visible)}')

with open(out_path) as _f:
    _html = _f.read()
# Last lines of Cell 9 — fix the display call
display(HTML(
    '<div id="syn-outer" style="width:100%;height:1400px;overflow:hidden;border:1px solid #ccc;border-radius:6px">'
    + _html + '</div>'
))












# ── Cell 11 ────────────────────────────────────────────────────────────────
from datetime import datetime, timezone
_utc_hour = datetime.now(timezone.utc).hour
if   _utc_hour <  6: EXPORT_TIME = "0000Z"
elif _utc_hour < 12: EXPORT_TIME = "0600Z"
elif _utc_hour < 18: EXPORT_TIME = "1200Z"
else:                EXPORT_TIME = "1800Z"
print(f'UTC hour: {_utc_hour}  →  default export: {EXPORT_TIME}')
with open('output/synoptic_map.html', 'r', encoding='utf-8') as f:


    html = f.read()

new_fn = '''function synSavePNG() {
  var btn    = document.getElementById("btn-save-png");
  var status = document.getElementById("save-status");
  if (btn) { btn.disabled = true; btn.textContent = "Capturing..."; }
  if (status) status.textContent = "";

  var keys = Object.keys(window).filter(function(k){ return k.startsWith("map_"); });
  if (!keys.length) { if(status) status.textContent="Map not found"; if(btn) btn.disabled=false; return; }
  var MAP   = window[keys[0]];
  var mapEl = document.getElementById(keys[0]) || document.querySelector(".leaflet-container");
  if (!mapEl) { if(status) status.textContent="Map el not found"; if(btn) btn.disabled=false; return; }

  var hideEls = [
    mapEl.querySelector(".leaflet-control-container"),
    document.querySelector(".leaflet-control-layers"),
    document.querySelector(".leaflet-control-zoom"),
    document.querySelector(".leaflet-control-attribution"),
    document.getElementById("syn-ts-bar"),
    document.getElementById("syn-save-bar"),
    document.getElementById("syn-fs-btn")
  ].filter(Boolean);
  var prevVis = hideEls.map(function(el){ return el.style.visibility; });
  hideEls.forEach(function(el){ el.style.visibility = "hidden"; });

  var CENTER = [55, -104];
  var ZOOM   = 5;
  var TARGET_W = 1400;
  var TARGET_H = 1100;

  var origW = mapEl.style.width;
  var origH = mapEl.style.height;

  function restore() {
    mapEl.style.width  = origW;
    mapEl.style.height = origH;
    MAP.invalidateSize();
    if (btn) { btn.disabled = false; btn.textContent = "Save PNG"; }
  }

  mapEl.style.width  = TARGET_W + "px";
  mapEl.style.height = TARGET_H + "px";
  MAP.invalidateSize();

  setTimeout(function() {
    MAP.setView(CENTER, ZOOM, { animate: false });
    setTimeout(function() {
      html2canvas(mapEl, {
        useCORS: true, allowTaint: true,
        scale: 2, logging: false,
        width: TARGET_W, height: TARGET_H
      }).then(function(canvas) {
        hideEls.forEach(function(el, i){ el.style.visibility = prevVis[i]; });

        var cropH = canvas.height;
        var cropW = Math.min(Math.round(cropH * 8.5 / 11.0), canvas.width);
        var out = document.createElement("canvas");
        out.width  = cropW;
        out.height = cropH;
        var ctx2 = out.getContext("2d");
        ctx2.drawImage(canvas, 0, 0, cropW, cropH, 0, 0, cropW, cropH);

        // ── White out outside frame ────────────────────────────────────
        var MARGIN = 36;
        ctx2.fillStyle = "rgba(255,255,255,1.0)";
        ctx2.fillRect(0,              0,              cropW,  MARGIN);        // top
        ctx2.fillRect(0,              cropH - MARGIN, cropW,  MARGIN);        // bottom
        ctx2.fillRect(0,              0,              MARGIN, cropH);         // left
        ctx2.fillRect(cropW - MARGIN, 0,              MARGIN, cropH);         // right
        // ──────────────────────────────────────────────────────────────

        // ── Timestamp label box (bottom-left) ──────────────────────────
        var today  = new Date();
        var months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];
        var dows   = ["SUNDAY","MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY"];
        var dowStr  = dows[today.getUTCDay()];
        var dateStr = months[today.getUTCMonth()] + " " + String(today.getUTCDate()).padStart(2,"0") + " " + today.getUTCFullYear();
        var selEl   = document.getElementById("ts-select");
        var tsVal   = selEl ? selEl.value : "";
        var timeStr = tsVal ? tsVal.slice(2) : "1200Z";
        var lines   = ["SURFACE MAP", dowStr + " " + dateStr, timeStr];
        var fSize   = 36;
        var pad     = 24;
        var lineH   = fSize * 1.3;
        var boxH    = lines.length * lineH + pad * 2;
        ctx2.font   = fSize + "px Arial, sans-serif";
        var maxW    = Math.max.apply(null, lines.map(function(l){ return ctx2.measureText(l).width; }));
        var boxW    = maxW + pad * 2;
        var bx      = MARGIN;
        var by      = cropH - MARGIN - boxH;
        ctx2.fillStyle = "rgba(255,255,255,0.88)";
        ctx2.fillRect(bx, by, boxW, boxH);
        ctx2.strokeStyle = "#1a4a8a";
        ctx2.lineWidth = 3;
        ctx2.strokeRect(bx, by, boxW, boxH);
        ctx2.fillStyle    = "#1a2030";
        ctx2.textBaseline = "top";
        ctx2.textAlign    = "center";
        var centerX = bx + boxW / 2;
        lines.forEach(function(line, i) {
          ctx2.font = fSize + "px Arial, sans-serif";
          ctx2.fillText(line, centerX, by + pad + i * lineH);
        });
// ── Frame border only ─────────────────────────────────────────
        var MARGIN = 36;
        ctx2.strokeStyle = "#1a2030";
        ctx2.lineWidth   = 2;
        ctx2.strokeRect(MARGIN, MARGIN, cropW - MARGIN * 2, cropH - MARGIN * 2);
        // ──────────────────────────────────────────────────────────────

// ── Corner lat/lon labels ─────────────────────────────────────
        var SC   = 2;
        var tlLL = MAP.containerPointToLatLng([MARGIN/SC,            MARGIN/SC]);
        var trLL = MAP.containerPointToLatLng([TARGET_W - MARGIN/SC, MARGIN/SC]);
        var blLL = MAP.containerPointToLatLng([MARGIN/SC,            TARGET_H - MARGIN/SC]);
        var brLL = MAP.containerPointToLatLng([TARGET_W - MARGIN/SC, TARGET_H - MARGIN/SC]);
        function fmtLat(v){ return Math.abs(v).toFixed(1)+(v>=0?"°N":"°S"); }
        function fmtLon(v){ return Math.abs(v).toFixed(1)+(v>=0?"°E":"°W"); }
        ctx2.font         = "18px Arial, sans-serif";
        ctx2.fillStyle    = "#1a2030";
        ctx2.textBaseline = "middle";
        var LAT_PAD = 30;
        [{ll:tlLL,x:MARGIN/2,y:MARGIN+LAT_PAD,r:-Math.PI/2},
         {ll:blLL,x:MARGIN/2,y:cropH-MARGIN-LAT_PAD,r:-Math.PI/2},
         {ll:trLL,x:cropW-MARGIN/2,y:MARGIN+LAT_PAD,r:Math.PI/2},
         {ll:brLL,x:cropW-MARGIN/2,y:cropH-MARGIN-LAT_PAD,r:Math.PI/2}
        ].forEach(function(p){
          ctx2.save(); ctx2.translate(p.x,p.y); ctx2.rotate(p.r);
          ctx2.textAlign="center"; ctx2.fillText(fmtLat(p.ll.lat),0,0); ctx2.restore();
        });
        var LON_PAD = 15;
        ctx2.textAlign="left";
        ctx2.fillText(fmtLon(tlLL.lng), MARGIN+LON_PAD,            MARGIN/2);
        ctx2.fillText(fmtLon(blLL.lng), MARGIN+LON_PAD,            cropH-MARGIN/2);
        ctx2.textAlign="right";
        ctx2.fillText(fmtLon(trLL.lng), cropW-MARGIN-LON_PAD,      MARGIN/2);
        ctx2.fillText(fmtLon(brLL.lng), cropW-MARGIN-LON_PAD,      cropH-MARGIN/2);
        // ──────────────────────────────────────────────────────────────



        // ── Export timestamp (bottom-right) ───────────────────────────
        var expNow = new Date();
        var expStr = "Exported at: "
          + expNow.getUTCFullYear() + "/"
          + String(expNow.getUTCMonth()+1).padStart(2,"0") + "/"
          + String(expNow.getUTCDate()).padStart(2,"0") + " "
          + String(expNow.getUTCHours()).padStart(2,"0") + ":"
          + String(expNow.getUTCMinutes()).padStart(2,"0") + ":"
          + String(expNow.getUTCSeconds()).padStart(2,"0") + "Z";
        ctx2.font         = "8px Arial, sans-serif";
        ctx2.fillStyle    = "#555555";
        ctx2.textBaseline = "middle";
        ctx2.textAlign    = "right";
        var lonLabelWidth = ctx2.measureText(fmtLon(brLL.lng)).width;
        ctx2.fillText(expStr, cropW - MARGIN - LON_PAD - lonLabelWidth - 60, cropH - MARGIN/2);
        // ──────────────────────────────────────────────────────────────


        // ── Build filename surface_plot_YYYYMMDDHHZ.png ─────────────────────
        var now = new Date();
        var yyyy = now.getUTCFullYear();
        var mm   = String(now.getUTCMonth()+1).padStart(2,"0");
        var dd   = String(now.getUTCDate()).padStart(2,"0");
        var selEl2  = document.getElementById("ts-select");
        var tsVal2  = selEl2 ? selEl2.value : "";
        // Extract HH from the timestamp value shown in the bottom-left box (e.g. "2024010112Z" → "12")
        // tsVal2 format: "131200Z" → strip Z, take chars at position 2-3 = "12"
        var tsStripped = tsVal2.replace(/Z$/i, "");
        var hh = tsStripped.length >= 4 ? tsStripped.slice(-4, -2) : "12";

        var name = "surface_plot_" + yyyy + mm + dd + hh + "Z-" + (window._synMetarPNG ? "no_contour" : "with_contour") + ".png";

        var link = document.createElement("a");
        link.download = name;
        link.href = out.toDataURL("image/png");
        link.click();
        restore();
        if (status) { status.textContent = "Saved!"; setTimeout(function(){ status.textContent = ""; }, 3000); }

      }).catch(function(e) {
        hideEls.forEach(function(el, i){ el.style.visibility = prevVis[i]; });
        restore();
        if (status) status.textContent = "Failed: " + e.message;
      });
    }, 300);
  }, 200);
}'''

# ── Replace synSavePNG by brace matching ──────────────────────────────
new_fn = new_fn.replace('"{EXPORT_TIME}"', f'"{EXPORT_TIME}"')
start = html.find('function synSavePNG() {')
if start == -1:
    print('ERROR: synSavePNG not found — run Cell 9 first')
else:
    depth = 0
    i = start
    while i < len(html):
        if html[i] == '{': depth += 1
        elif html[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1
    html = html[:start] + new_fn + html[end:]
    print('synSavePNG replaced')

# ── Hide contours and H/L ─────────────────────────────────────────────
html = html.replace('var _synShowSlp = true;', 'var _synShowSlp = false;')
html = html.replace('var _synShowHL  = true;', 'var _synShowHL  = false;')

# ── Inject synExport1200Z ─────────────────────────────────────────────
import re
_body_idx = html.rfind('</body>')
_inject_idx = html.rfind('<script>', 0, _body_idx)
while _inject_idx != -1 and 'synExport1200Z' not in html[_inject_idx:_body_idx]:
    _inject_idx = html.rfind('<script>', 0, _inject_idx)
if _inject_idx != -1 and 'synExport1200Z' in html[_inject_idx:_body_idx]:
    html = html[:_inject_idx] + html[_body_idx:]
if True:  # always inject

    _btn_bg  = "#ffd8a8" if EXPORT_TIME in ("1800Z","0000Z") else "#c8dff4"
    _btn_bdr = "#a85c00" if EXPORT_TIME in ("1800Z","0000Z") else "#1a4a8a"
    _btn_clr = "#5c2e00" if EXPORT_TIME in ("1800Z","0000Z") else "#1a3a6a"
    export_js = '''<script>
function synExport1200Z() {
  var sel = document.getElementById("ts-select");
  if (sel) {
    var target = "{EXPORT_TIME}";
    var opts = Array.from(sel.options).filter(function(o){ return o.value.indexOf(target) !== -1; });
    if (!opts.length) {
      var hh = target.replace("Z","");
      opts = Array.from(sel.options).filter(function(o){ return o.value.slice(-4,-2) === hh || o.value.slice(2,4) === hh; });
    }
    if (opts.length) { sel.value = opts[opts.length-1].value; synUpdateTS(sel.value); }
  }
  setTimeout(synSavePNG, 800);
}
function synExportCurrent() {
  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});
  var MAP = keys.length ? window[keys[0]] : null;
  if (!_synShowSlp && MAP) { _synShowSlp = true; _synSlpLayer.addTo(MAP); var btn=document.getElementById("btn-slp"); if(btn){btn.textContent="Isobars ✓";btn.style.background="#e8f0fe";} }
  if (!_synShowHL  && MAP) { _synShowHL  = true; _synHLLayer.addTo(MAP);  var btn2=document.getElementById("btn-hl");  if(btn2){btn2.textContent="H/L ✓";btn2.style.background="#e8f0fe";} }
  var styleEl = document.getElementById("syn-wx-style");
  if (styleEl) styleEl.textContent = "";
  if (_synSvgMode === "wmo") { _synSvgMode = "colour"; var btn3=document.getElementById("btn-svg"); if(btn3){btn3.textContent="Stn ✓";btn3.style.background="#e8f0fe";btn3.style.color="#1a3a6a";btn3.style.borderColor="#aaa";} var _sel=document.getElementById("ts-select"); if(_sel&&_sel.value) synUpdateTS(_sel.value); }
  setTimeout(synSavePNG, 400);
}
function synExportCurrentMetar() {
  var hadSlp = _synShowSlp, hadHL = _synShowHL;
  if (hadSlp) { _synShowSlp = false; _synSlpLayer.remove(); }
  if (hadHL)  { _synShowHL  = false; _synHLLayer.remove();  }
  var styleEl = document.getElementById("syn-wx-style");
  if (!styleEl) { styleEl = document.createElement("style"); styleEl.id = "syn-wx-style"; document.head.appendChild(styleEl); }
  var hadWxHidden = styleEl.textContent.indexOf("syn-wx-box") !== -1;
  styleEl.textContent = ".syn-wx-box { display: none !important; }";
  window._synMetarPNG = true;
  setTimeout(function() {
    synSavePNG();
    setTimeout(function() {
      if (hadSlp) { _synShowSlp = true; _synSlpLayer.addTo(MAP); }
      if (hadHL)  { _synShowHL  = true; _synHLLayer.addTo(MAP);  }
      if (!hadWxHidden) styleEl.textContent = "";
      window._synMetarPNG = false;
    }, 3000);
  }, 200);
}
function synToggleSvgForMetar(hide, keys) {
  var MAP = window[keys[0]];
  if (hide) {
    if (_synStnLayer) MAP.removeLayer(_synStnLayer);
  } else {
    if (_synShowSvg && _synStnLayer) _synStnLayer.addTo(MAP);
  }
}
function synExportMetar() {
  var hadSlp = _synShowSlp, hadHL = _synShowHL;
  if (hadSlp) { _synShowSlp = false; _synSlpLayer.remove(); }
  if (hadHL)  { _synShowHL  = false; _synHLLayer.remove();  }
  var styleEl = document.getElementById("syn-wx-style");
  if (!styleEl) { styleEl = document.createElement("style"); styleEl.id = "syn-wx-style"; document.head.appendChild(styleEl); }
  var hadWxHidden = styleEl.textContent.indexOf("syn-wx-box") !== -1;
  styleEl.textContent = ".syn-wx-box { display: none !important; }";
  window._synMetarPNG = true;
  setTimeout(function() {
    synSavePNG();
    setTimeout(function() {
      if (hadSlp) { _synShowSlp = true; _synSlpLayer.addTo(MAP); }
      if (hadHL)  { _synShowHL  = true; _synHLLayer.addTo(MAP);  }
      if (!hadWxHidden) styleEl.textContent = "";
      window._synMetarPNG = false;
    }, 3000);
  }, 200);
}
</script>
<div style="position:fixed;top:10px;right:10px;z-index:10002;display:flex;flex-direction:column;gap:6px;">
  <button onclick="synExport1200Z()" style="font-family:Courier New,monospace;font-size:12px;
    padding:5px 12px;background:{BTN_BG};border:1px solid {BTN_BDR};border-radius:5px;
    color:{BTN_CLR};cursor:pointer;font-weight:bold;">&#9928; Export {EXPORT_TIME} Analysis PNG</button>
  <button onclick="synExportMetar()" style="font-family:Courier New,monospace;font-size:12px;
    padding:5px 12px;background:#f4e8c8;border:1px solid #9a6a00;border-radius:5px;
    color:#4a3000;cursor:pointer;font-weight:bold;">&#128225; Export {EXPORT_TIME} METAR PNG</button>
  <button onclick="synExportCurrent()" style="font-family:Courier New,monospace;font-size:12px;
    padding:5px 12px;background:#d4f4c8;border:1px solid #1a6a2a;border-radius:5px;
    color:#1a3a1a;cursor:pointer;font-weight:bold;">&#9200; Export Current Timestep PNG</button>
  <button onclick="synExportCurrentMetar()" style="font-family:Courier New,monospace;font-size:12px;
    padding:5px 12px;background:#e8f4e8;border:1px solid #2a7a3a;border-radius:5px;
    color:#1a4a1a;cursor:pointer;font-weight:bold;">&#128225; Export Current Timestep METAR PNG</button>
  <button onclick="synShowRunPanel()" style="font-family:Courier New,monospace;font-size:12px;
    padding:5px 12px;background:#f0e8f8;border:1px solid #6a2a9a;border-radius:5px;
    color:#3a006a;cursor:pointer;font-weight:bold;"><span id="gha-run-btn-text">&#9881; Run Script Now</span><span id="gha-last-run" style="font-size:9px;font-weight:normal;color:#9a6acc;margin-left:6px;"></span></button>
  <div id="gha-panel" style="display:none;flex-direction:row;align-items:center;gap:6px;padding:5px 10px;
    background:#faf8ff;border:1px solid #9a6acc;border-radius:5px;">
    <span style="color:#555;font-size:11px;font-family:Courier New,monospace;">PIN</span>
    <input id="gha-pin" type="password" maxlength="4" placeholder="····"
      onkeydown="if(event.key==='Enter')synTriggerGHA()"
      style="width:52px;font-family:Courier New,monospace;font-size:12px;padding:3px 5px;
      border:1px solid #9a6acc;border-radius:3px;text-align:center;"/>
    <button onclick="synTriggerGHA()" style="padding:3px 10px;background:#7a2acc;border:none;
      border-radius:3px;color:white;cursor:pointer;font-family:Courier New,monospace;font-size:11px;font-weight:bold;">&#9889; Run</button>
    <span id="gha-status" style="color:#555;font-size:10px;font-family:Courier New,monospace;"></span>
  </div>
  <div id="gha-progress" style="display:none;flex-direction:column;gap:3px;padding:6px 10px;
    background:#faf8ff;border:1px solid #9a6acc;border-radius:5px;">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <b style="color:#3a006a;font-family:Courier New,monospace;font-size:11px;">&#128640; Workflow Progress</b>
      <span id="gha-run-status" style="color:#888;font-size:10px;font-family:Courier New,monospace;"></span>
    </div>
    <div style="background:#e8e0f0;border-radius:3px;height:6px;overflow:hidden;">
      <div id="gha-bar" style="height:100%;width:0%;background:linear-gradient(90deg,#7a2acc,#a855f7);border-radius:3px;transition:width 0.6s ease;"></div>
    </div>
    <div id="gha-steps" style="display:flex;flex-direction:column;gap:2px;font-family:Courier New,monospace;font-size:10px;"></div>
  </div>
</div>'''.replace('{BTN_BG}', _btn_bg).replace('{BTN_BDR}', _btn_bdr).replace('{BTN_CLR}', _btn_clr)



    export_js = export_js.replace('"{EXPORT_TIME}"', f'"{EXPORT_TIME}"')
    export_js = export_js.replace('{EXPORT_TIME}', EXPORT_TIME)
    html = html.replace('</body>', export_js + '</body>')
    print('Export 1200Z injected')
else:
    print('Export 1200Z already present')


# ── Disable all hover tooltips ────────────────────────────────────────

# 1. Station markers tooltip
html = html.replace(
    '}).bindPopup(d.popup,{maxWidth:280,closeButton:true}).bindTooltip(d.tip).addTo(_synStnLayer);',
    '}).bindPopup(d.popup,{maxWidth:280,closeButton:true}).addTo(_synStnLayer);'
)

# 2. Isobar contour lines tooltip
html = html.replace(
    '}).bindTooltip(Math.round(ct.level)+" ").addTo(_synSlpLayer);',
    '}).addTo(_synSlpLayer);'
)

# 3. H/L markers tooltip
html = html.replace(
    '}).bindTooltip(c.type).addTo(_synHLLayer);',
    '}).addTo(_synHLLayer);'
)

print('Tooltips disabled')

# ── Auto-trigger export on load ───────────────────────────────────────
with open('output/synoptic_map.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Saved')


# clear_output (no-op in headless mode)
with open('output/synoptic_map.html', encoding='utf-8') as f:
    _html = f.read()

display(HTML(
    f'<div style="font-family:Courier New,monospace;padding:8px 10px;'
    f'background:#f8f8f8;border:1px solid #ccc;border-radius:6px 6px 0 0;'
    f'display:flex;gap:10px;align-items:center;">'
    f'<button onclick="synExport1200Z()" style="font-size:13px;padding:7px 16px;'
    f'background:{_btn_bg};border:1px solid {_btn_bdr};border-radius:5px;'
    f'color:{_btn_clr};cursor:pointer;font-weight:bold;">&#9928; Export {EXPORT_TIME} Analysis PNG</button>'
    f'<button onclick="synExportMetar()" style="font-size:13px;padding:7px 16px;'
    f'background:#f4e8c8;border:1px solid #9a6a00;border-radius:5px;'
    f'color:#4a3000;cursor:pointer;font-weight:bold;">&#128225; Export {EXPORT_TIME} METAR PNG</button>'
    f'<button onclick="synExportCurrent()" style="font-size:13px;padding:7px 16px;'
    f'background:#d4f4c8;border:1px solid #1a6a2a;border-radius:5px;'
    f'color:#1a3a1a;cursor:pointer;font-weight:bold;">&#128247; Export Current PNG</button>'
    f'</div>'
    f'<div style="width:100%;height:1800px;border:1px solid #ccc;border-radius:0 0 6px 6px;overflow:hidden;">'
    + _html +
    f'</div>'
))



# auto-export removed — download triggered manually via button in the HTML

# -- Cell 8 - WMO station model as SVG string ---
import math

_CR = 0.14   # smaller station circle

# ── WX HIGHLIGHT BACKGROUND COLOURS ───────────────────────────────────────
_WX_BG = {
    'FZRA': '#cc2222',
    'FZDZ': '#cc2222',
    'TS':   '#ff00cc',
    'RA':   '#22aa44', 'SN': '#22aa44', 'DZ': '#22aa44',
    'SG':   '#22aa44', 'IC': '#22aa44', 'GR': '#22aa44',
    'GS':   '#22aa44', 'PL': '#22aa44', 'UP': '#22aa44',
    'BR':   '#dddd00', 'FG': '#dddd00',
    'HZ':   '#888888', 'FU': '#dd7700', 'DU': '#888888',
    'SA':   '#888888', 'VA': '#888888',
}

# wx codes that fade at high visibility
_WX_VIS_FADE = {'RA', 'SN', 'DZ', 'SG', 'IC', 'GR', 'GS', 'PL', 'UP'}




def _wx_bg_color(wx_str):
    """Return (bg_colour, fade_at_high_vis) for a wx string, or (None, False)."""
    if not wx_str:
        return None, False
    s = wx_str.upper()
    for key in ('FZRA', 'FZDZ', 'TS', 'RA', 'SN', 'DZ', 'SG', 'IC', 'GR',
                'GS', 'PL', 'UP', 'BR', 'FG', 'HZ', 'FU',
                'DU', 'SA', 'VA'):
        if key in s:
            return _WX_BG[key], (key in _WX_VIS_FADE)
    return None, False
# ──────────────────────────────────────────────────────────────────────────


# ── FEATHER ANGLE CONTROL ──────────────────────────────────────────────────
# Angle of feathers relative to staff (degrees).
# 90  = perpendicular to staff (standard WMO)
# >90 = feathers tilt AWAY from circle (toward tip)
# <90 = feathers tilt TOWARD circle
FEATHER_ANGLE = 110   # ← change this value to adjust feather angle

# Side of feather: +1 = right side of staff (looking from base to tip)
#                  -1 = left side  (standard WMO)
FEATHER_SIDE = +1    # ← change to +1 to flip to right side
# ──────────────────────────────────────────────────────────────────────────


def cloud_circle_svg(cx, cy, R, oktas):
    lw = max(0.9, R * 0.13)
    s = []
    if oktas == 9:  # VV: full black + white X
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="black" stroke="black" stroke-width="{lw}"/>')
        s.append(f'<line x1="{cx-R*.55:.2f}" y1="{cy-R*.55:.2f}" x2="{cx+R*.55:.2f}" y2="{cy+R*.55:.2f}" stroke="white" stroke-width="{lw*.85:.2f}"/>')
        s.append(f'<line x1="{cx+R*.55:.2f}" y1="{cy-R*.55:.2f}" x2="{cx-R*.55:.2f}" y2="{cy+R*.55:.2f}" stroke="white" stroke-width="{lw*.85:.2f}"/>')
        return ''.join(s)
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="white" stroke="white" stroke-width="{lw*3:.2f}"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="white" stroke="black" stroke-width="{lw}"/>')
    if oktas <= 0:
        return ''.join(s)
    if oktas >= 8:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="black" stroke="black" stroke-width="{lw}"/>')
        return ''.join(s)
    if oktas == 2:
        s.append(f'<path d="M{cx},{cy} L{cx},{cy-R:.2f} A{R:.2f},{R:.2f} 0 0,1 {cx+R:.2f},{cy} Z" fill="black"/>')
    elif oktas == 4:
        s.append(f'<path d="M{cx},{cy} L{cx},{cy-R:.2f} A{R:.2f},{R:.2f} 0 1,1 {cx},{cy+R:.2f} Z" fill="black"/>')
    elif oktas == 6:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="black" stroke="black" stroke-width="{lw}"/>')
        s.append(f'<path d="M{cx},{cy} L{cx-R:.2f},{cy} A{R:.2f},{R:.2f} 0 0,1 {cx},{cy-R:.2f} Z" fill="white"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="black" stroke-width="{lw}"/>')
    return ''.join(s)


def wind_barb_svg(cx, cy, R, wind_dir, wind_spd, wind_gust, S):
    """
    WMO wind barb using SVG transform rotate.
    Draws canonical FROM-NORTH barb, then rotates by wind_dir degrees.

    Feather direction and angle controlled by module-level constants:
      FEATHER_SIDE  : -1 = left (WMO standard), +1 = right
      FEATHER_ANGLE : degrees from staff (90=perpendicular, >90=tilts away from circle)
    """
    if wind_dir is None or wind_spd is None:
        return ''
    if wind_spd < 3:
        return (f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{R*1.5:.2f}" '
                f'fill="none" stroke="black" stroke-width="1"/>')

    sl        = S * 1.0    # staff length
    blen      = S * 0.30    # full barb (10-kt feather) length
    blen_penn = S * 0.45    # pennant (50-kt triangle) width
    bspc      = S * 0.115   # spacing between barbs along staff
    lw        = max(0.9, S * 0.038)  # line width

    staff_base_y = -R
    staff_tip_y  = -(R + sl)

    # Feather x-endpoint and y-tilt from FEATHER_SIDE and FEATHER_ANGLE
    # x goes FEATHER_SIDE direction; y tilt = tan(angle-90) * blen toward tip (-y)
    fx_full = FEATHER_SIDE * blen
    fx_half = FEATHER_SIDE * blen * 0.5
    # tilt: >90 deg tilts feather end toward tip (negative y = up)
    tilt = math.tan(math.radians(FEATHER_ANGLE - 90)) * blen

    spd = int(round(wind_spd / 5.0)) * 5
    pn  = spd // 50;  spd -= pn * 50
    fu  = spd // 10;  spd -= fu * 10
    ha  = spd //  5

    parts = []
    parts.append(f'<line x1="0" y1="{staff_base_y:.2f}" x2="0" y2="{staff_tip_y:.2f}" '
             f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round" '
             f'paint-order="stroke" style="paint-order:stroke;stroke:black;'
             f'-webkit-text-stroke:white {lw*3:.2f}px"/>')

    pos = 0.0

    if pn == 0 and fu == 0 and ha == 1:
        hy = staff_tip_y + 0.28 * sl
        parts.append(f'<line x1="0" y1="{hy:.2f}" x2="{fx_half:.2f}" y2="{hy - tilt*0.5:.2f}" '
                     f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>')
    else:
        for _ in range(pn):  # 50-kt pennants
            ay  = staff_tip_y + pos
            by2 = staff_tip_y + pos + bspc * 2
            pts = f'0,{ay:.2f} {fx_full:.2f},{ay - tilt:.2f} 0,{by2:.2f}'
            parts.append(f'<polygon points="{pts}" fill="black"/>')
            pos += bspc * 1.5
        for _ in range(fu):  # 10-kt full barbs
            fy = staff_tip_y + pos
            parts.append(f'<line x1="0" y1="{fy:.2f}" x2="{fx_full:.2f}" y2="{fy - tilt:.2f}" '
                         f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>')
            pos += bspc
        for _ in range(ha):  # 5-kt half barbs
            hy = staff_tip_y + pos
            parts.append(f'<line x1="0" y1="{hy:.2f}" x2="{fx_half:.2f}" y2="{hy - tilt*0.5:.2f}" '
                         f'stroke="black" stroke-width="{lw:.2f}" stroke-linecap="round"/>')
            pos += bspc

    inner = ''.join(parts)
    return (f'<g transform="translate({cx:.2f},{cy:.2f}) rotate({wind_dir:.1f})">'
            f'{inner}</g>')

def pressure_tendency_svg(cx, cy, R, tendency, S):
    """
    WMO pressure tendency characteristic symbol, drawn to the right of the
    station circle, vertically centred on the SLP label row.

    tendency codes (WMO):
      0 = rising then falling  (∧  inverted-V)
      1 = rising then steady   (⌐)
      2 = rising               (/)
      3 = falling then rising  - not listed but keep slot
      4 = steady               (—)
      5 = falling then rising  (V)
      6 = falling then steady  (∟)
      7 = falling              (\\)
      8 = steady then falling  (not common, map to 7)

    Also accepts string keys: 'rising', 'falling', 'steady',
      'rising_falling', 'falling_rising', 'rising_steady', 'falling_steady'
    """
    _map = {
        'rising':          2,
        'falling':         7,
        'steady':          4,
        'rising_falling':  0,
        'falling_rising':  5,
        'rising_steady':   1,
        'falling_steady':  6,
    }
    if isinstance(tendency, str):
        tendency = _map.get(tendency.lower(), None)
    if tendency is None:
        return ''

    lw  = max(0.9, S * 0.042)
    # position: right of circle, on the SLP-label row (slightly below centre)
    ox  = cx + R + S * 0.09 + S * 0.52   # shifted right to leave room for change amount
    slp_y = cy - R * 0.6 - 7         # matches slp_label y
    oy  = slp_y + S * 0.55           # one row below SLP label

    arm = S * 0.22    # half-width of symbol
    rise = S * 0.20   # vertical rise of symbol

    def line(x1, y1, x2, y2):
        return (f'<line x1="{ox+x1:.2f}" y1="{oy+y1:.2f}" '
                f'x2="{ox+x2:.2f}" y2="{oy+y2:.2f}" '
                f'stroke="black" stroke-width="{lw:.2f}" '
                f'stroke-linecap="round" stroke-linejoin="round"/>')

    parts = []

    if tendency == 2:        # Rising  /
        parts.append(line(-arm,  rise*0.5, arm, -rise*0.5))

    elif tendency == 7:      # Falling  \
        parts.append(line(-arm, -rise*0.5, arm,  rise*0.5))

    elif tendency == 4:      # Steady  —
        parts.append(line(-arm, 0, arm, 0))

    elif tendency == 0:      # Rising then falling  ∧
        parts.append(line(-arm,  rise*0.5,   0, -rise*0.5))
        parts.append(line(  0, -rise*0.5,  arm,  rise*0.5))

    elif tendency == 5:      # Falling then rising  V
        parts.append(line(-arm, -rise*0.5,   0,  rise*0.5))
        parts.append(line(  0,  rise*0.5,  arm, -rise*0.5))

    elif tendency == 1:      # Rising then steady  ⌐
        parts.append(line(-arm,  rise*0.5,   0, -rise*0.5))   # rising stroke
        parts.append(line(  0,  -rise*0.5, arm, -rise*0.5))   # horizontal tail

    elif tendency == 6:      # Falling then steady  ∟
        parts.append(line(-arm, -rise*0.5,   0,  rise*0.5))   # falling stroke
        parts.append(line(  0,   rise*0.5, arm,  rise*0.5))   # horizontal tail

    return ''.join(parts)


def station_model_svg(d, S=34, wmo_style=False):
    """Full WMO station model SVG. wmo_style=True skips colour boxes."""
    PAD = S * 1.2
    W   = S * 3 + PAD * 2
    H   = S * 3 + PAD * 2
    cx  = W / 2
    cy  = H / 2
    R   = S * _CR
    fs  = globals().get('FONT_SCALE', max(7, int(S * 0.36)))
    off = R + S * 0.09
    hide_labels = d.get('icao', '').upper() in {'CZPC', 'CWGM'}

    parts = []
    has_cloud = d.get('has_sky_obs', False)
    if has_cloud:
        parts.append(cloud_circle_svg(cx, cy, R, d['oktas']))
    else:
        # No sky sensor — draw black triangle, same centre as circle
        th = R * 1.6
        tx1, ty1 = cx,        cy - th
        tx2, ty2 = cx - th,   cy + th * 0.65
        tx3, ty3 = cx + th,   cy + th * 0.65
        parts.append(f'<polygon points="{tx1:.2f},{ty1:.2f} {tx2:.2f},{ty2:.2f} {tx3:.2f},{ty3:.2f}" fill="black" stroke="none"/>')
    parts.append(wind_barb_svg(cx, cy, R,
                               d['wind_dir'], d['wind_spd'],
                               d.get('wind_gust', 0), S))

    def txt(x, y, text, anchor='end', bold=False, size=None):
        sz = size or fs
        fw = 'bold' if bold else 'normal'
        return (f'<text x="{x:.1f}" y="{y:.1f}" '
                f'text-anchor="{anchor}" dominant-baseline="central" '
                f'font-size="{sz}px" font-weight="{fw}" '
                f'font-family="Courier New,monospace" fill="black" '
                f'paint-order="stroke" stroke="white" '
                f'stroke-width="2" stroke-linejoin="round">'
                f'{text}</text>')

    if not hide_labels:
        if d['temp'] is not None:
            parts.append(txt(cx - off, cy - R * 0.6 - 6, str(d['temp']), bold=False))
        v  = d['vis']
        vs = (str(int(v))  if v is not None and v >= 10    else
              str(int(v))  if v is not None and v % 1 == 0 else
              f'{v:.1f}'   if v is not None                else None)
        wx = ' '.join(x for x in [vs, d['weather'] or None] if x)
        if wx:
            _wx_x  = cx - off - 4
            _wx_y  = cy
            _bg, _fade = _wx_bg_color(d.get('weather', ''))
            if _bg and not wmo_style:
                _char_w = (fs * 0.62)
                _tw     = len(wx) * _char_w
                _px, _py = 3, 2
                if _fade and v is not None and v > 6:
                    _opacity = 0.20
                else:
                    _opacity = 0.95
                parts.append(
                    f'<rect '
                    f'x="{_wx_x - _tw - _px:.1f}" '
                    f'y="{_wx_y - fs * 0.5 - _py:.1f}" '
                    f'width="{_tw + _px * 2:.1f}" '
                    f'height="{fs + _py * 2:.1f}" '
                    f'rx="2" fill="{_bg}" opacity="{_opacity}" class="syn-wx-box"/>'
                )
            parts.append(txt(_wx_x, _wx_y, wx, bold=False))
        if d['dew'] is not None:
            parts.append(txt(cx - off, cy + R * 0.6 + 6, str(d['dew'])))
    if d['slp_label']:
        parts.append(txt(cx + off, cy - R * 0.6 - 7, d['slp_label'], anchor='start'))
    tendency = d.get('tendency')
    pressure_change = d.get('pressure_change')
    if not hide_labels and tendency is not None:
        tend_y = cy - R * 0.6 - 7 + S * 0.55
        has_number = tendency != 'steady' and pressure_change is not None
        if has_number:
            pc_str = ('+' if pressure_change > 0 else '-' if pressure_change < 0 else '') + str(abs(pressure_change))
            parts.append(txt(cx + off, tend_y, pc_str, anchor='start'))
            parts.append(pressure_tendency_svg(cx + off, cy, R, tendency, S))
        else:
            # no number — shift symbol left to sit flush with SLP label column
            parts.append(pressure_tendency_svg(cx + off - S * 0.52, cy, R, tendency, S))

    if d['lowest_sig'] and d['lowest_sig']['height'] <= 120:
        _cb = math.ceil(d['lowest_sig']['height'] / 10)
        _cb_str = str(_cb)
        _cb_x = cx
        _cb_y = cy + R + fs * 0.9
        if not wmo_style:
            _ceil_color = '#880088' if d['lowest_sig']['height'] <= 60 else '#8B4513'
            _char_w = fs * 0.62
            _tw = len(_cb_str) * _char_w
            _px, _py = 3, 2
            parts.append(
                f'<rect '
                f'x="{_cb_x - _tw/2 - _px:.1f}" '
                f'y="{_cb_y - fs * 0.5 - _py:.1f}" '
                f'width="{_tw + _px * 2:.1f}" '
                f'height="{fs + _py * 2:.1f}" '
                f'rx="2" fill="{_ceil_color}" opacity="0.82" class="syn-wx-box"/>'
            )
        parts.append(txt(_cb_x, _cb_y, _cb_str, anchor='middle'))
    _name_y = cy + R + fs * 0.9 + fs * 1.2
    parts.append(txt(cx, _name_y, d['icao'][-3:], anchor='middle'))
    return (f'<svg width="{W:.0f}" height="{H:.0f}" '
            f'viewBox="0 0 {W:.2f} {H:.2f}" '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'style="overflow:visible">'
            + ''.join(parts) + '</svg>'), W, H


def flight_cat_color(d):
    return {'VFR': '#22aa44', 'MVFR': '#2244cc',
            'IFR': '#cc2222', 'LIFR': '#880088'}.get(d.get('flt_cat', ''), '#888888')


# ── DEMO: render 27025KT and a few cardinal directions ────────────────────
print(f'Colored - Station model SVG ready  '
      f'(FEATHER_SIDE={FEATHER_SIDE}, FEATHER_ANGLE={FEATHER_ANGLE}°)')

# -- WX Sample Display --
# Renders one station model per WX code using station_model_svg()
# Paste this cell after Cell 8 in your notebook.

import math

WX_SAMPLES = [
    # code,  vis,  temp, dew,  wind_dir, wind_spd, label
    ('FZRA',  1,   -2,   -4,   220, 15, 'FZRA – freezing rain'),
    ('FZDZ',  2,   -1,   -3,   200, 10, 'FZDZ – freezing drizzle'),
    ('TS',    5,   18,   14,   270, 20, 'TS – thunderstorm'),
    ('RA',    4,   10,    8,   220, 15, 'RA – rain (low vis)'),
    ('RA',   15,   12,    9,   220, 15, 'RA – rain (high vis, faded)'),
    ('SN',    3,   -3,   -5,   300, 12, 'SN – snow'),
    ('DZ',    2,    8,    7,   180,  8, 'DZ – drizzle'),
    ('SG',    3,   -2,   -4,   310, 10, 'SG – snow grains'),
    ('IC',    4,  -10,  -12,    20,  5, 'IC – ice crystals'),
    ('GR',    5,   16,   12,   250, 25, 'GR – hail'),
    ('GS',    4,    5,    3,   230, 18, 'GS – small hail'),
    ('PL',    3,    0,   -2,   190, 14, 'PL – ice pellets'),
    ('UP',    2,    6,    4,   160, 10, 'UP – unknown precip'),
    ('BR',    3,   14,   12,   100,  5, 'BR – mist'),
    ('FG',    0,   10,   10,     0,  2, 'FG – fog (calm)'),
    ('FU',    5,   22,    8,    60, 12, 'FU – smoke'),
    ('HZ',    6,   28,   10,   330,  8, 'HZ – haze'),
    ('DU',    4,   30,    5,   350, 20, 'DU – widespread dust'),
    ('SA',    3,   32,    4,    10, 25, 'SA – sand'),
    ('VA',    4,   15,    8,   240, 18, 'VA – volcanic ash'),
]

S = 34  # station model scale

def make_station(code, vis, temp, dew, wind_dir, wind_spd):
    d = {
        'temp':        temp,
        'dew':         dew,
        'vis':         vis,
        'weather':     code,
        'wind_dir':    wind_dir,
        'wind_spd':    wind_spd,
        'wind_gust':   0,
        'oktas':       4,
        'has_sky_obs': True,
        'slp_label':   '132',
        'tendency':    None,
        'pressure_change': None,
        'lowest_sig':  None,
        'icao':        'WX' + code[:3],
    }
    svg, w, h = station_model_svg(d, S=S)
    return svg, w, h

COLS = 5
items = []
for (code, vis, temp, dew, wdir, wspd, label) in WX_SAMPLES:
    svg, w, h = make_station(code, vis, temp, dew, wdir, wspd)
    cell = f'''
    <div style="display:flex;flex-direction:column;align-items:center;
                font-family:'Courier New',monospace;font-size:10px;
                color:#555;gap:2px;">
      {svg}
      <span style="max-width:{int(w)}px;text-align:center;line-height:1.3;">{label}</span>
    </div>'''
    items.append(cell)

rows_html = ''
for i in range(0, len(items), COLS):
    chunk = items[i:i+COLS]
    rows_html += f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px;">{"".join(chunk)}</div>'

html = f'''
<div style="background:#f8f8f8;padding:12px;border-radius:6px;
            border:1px solid #ddd;display:inline-block;">
  <div style="font-family:\'Courier New\',monospace;font-size:12px;
              font-weight:bold;margin-bottom:8px;color:#333;">
    WX Code Samples
  </div>
  {rows_html}
</div>
'''

display(HTML(html))

#color coding WX and Cloud
# 1. No countour
# -- Cell 9 - Interactive Folium map with OSM tiles ---
import folium
from folium import Element
import json as _json
import numpy as np
from matplotlib import pyplot as plt
import math as _math


print('Creating color coded map')

# -- build the map ---
center_lat = 56
center_lon = -114
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles=None,
    prefer_canvas=True
)

# tile layers — Blank added last so Leaflet selects it as default
folium.TileLayer(
    tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    attr='CartoDB Positron', name='White (CartoDB)', max_zoom=19
).add_to(m)
folium.TileLayer(tiles='OpenStreetMap', name='OpenStreetMap', max_zoom=19).add_to(m)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    attr='ESRI World Topo', name='ESRI Topo', max_zoom=19
).add_to(m)
folium.TileLayer(
    tiles='about:blank',
    attr='Blank', name='Blank (borders only)', max_zoom=19
).add_to(m)
# white background + force blank as default on load
m.get_root().html.add_child(Element(
    '<style>.leaflet-container{background:#ffffff!important;}</style>\n'
    '<script>\n'
    '(function(){\n'
    '  function initBlank(){\n'
    '    var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    if(!keys.length){setTimeout(initBlank,200);return;}\n'
    '    var MAP=window[keys[0]];\n'
    '    var blankLayer=null;\n'
    '    var others=[];\n'
    '    MAP.eachLayer(function(l){\n'
    '      if(l instanceof L.TileLayer){\n'
    '        if(l.options.name==="Blank (borders only)" || (l._url&&l._url==="about:blank")){\n'
    '          blankLayer=l;\n'
    '        } else {\n'
    '          others.push(l);\n'
    '        }\n'
    '      }\n'
    '    });\n'
    '    others.forEach(function(l){MAP.removeLayer(l);});\n'
    '    if(blankLayer) blankLayer.addTo(MAP);\n'
    '  }\n'
    '  function tryInitBlank(){\n'
    '    var keys=Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    if(!keys.length){setTimeout(tryInitBlank,200);return;}\n'
    '    var MAP=window[keys[0]];\n'
    '    // also fire on every layeradd to catch late-loading tiles\n'
    '    MAP.on("layeradd", function(){\n'
    '      MAP.eachLayer(function(l){\n'
    '        if(l instanceof L.TileLayer && l._url && l._url!=="about:blank"){\n'
    '          MAP.removeLayer(l);\n'
    '        }\n'
    '      });\n'
    '    });\n'
    '    initBlank();\n'
    '  }\n'
    '  if(document.readyState==="complete"){setTimeout(tryInitBlank,100);}\n'
    '  else{window.addEventListener("load",function(){setTimeout(tryInitBlank,100);});}\n'
    '})();\n'
    '</script>\n'
))

# ---- SLP CONTOURS --------------------------------------------------------
slp_fg = folium.FeatureGroup(name='SLP Isobars', show=None)

if slp_grid is not None:
    glon, glat = np.meshgrid(lon_vec, lat_vec)
    slp_min = np.floor(slp_grid.min() / SLP_INTERVAL) * SLP_INTERVAL
    slp_max = np.ceil(slp_grid.max()  / SLP_INTERVAL) * SLP_INTERVAL
    levels  = np.arange(slp_min, slp_max + SLP_INTERVAL, SLP_INTERVAL)
    fig_c, ax_c = plt.subplots(figsize=(1, 1))
    cs = ax_c.contour(glon, glat, slp_grid, levels=levels)
    plt.close(fig_c)
    for lvl_idx, lvl in enumerate(cs.levels):
        is_major = (int(lvl) % 20 == 0)
        weight   = 2.5 if is_major else (1.4 if int(lvl)%8==0 else 0.7)
        opacity  = 0.95 if is_major else (0.65 if int(lvl)%8==0 else 0.40)
        for coords in cs.allsegs[lvl_idx]:
            if len(coords) < 2:
                continue
            geo_coords = [[float(c[0]), float(c[1])] for c in coords]
            feature = {
                'type': 'Feature',
                'geometry': {'type': 'LineString', 'coordinates': geo_coords},
                'properties': {'level': float(lvl)}
            }
            folium.GeoJson(
                feature,
                style_function=lambda f: {
                'color': '#000000', 'weight': 1, 'opacity': 1.0
                },
                tooltip=folium.Tooltip(f'{int(lvl)} ')
            ).add_to(slp_fg)
            if int(lvl) % 4 == 0 and len(coords) > 4:
                mid = coords[len(coords)//2]
                folium.Marker(
                    location=[float(mid[1]), float(mid[0])],
                    icon=folium.DivIcon(
                        html=(f'<div style="font-size:9px;font-weight:900;color:#1a3a6a;'
                              f'font-family:Courier New,monospace;white-space:nowrap;'
                              f'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,'
                              f'1px -1px 0 #fff,-1px 1px 0 #fff;">'
                              f'{int(lvl)}</div>'),
                        icon_size=(32, 14), icon_anchor=(16, 7)
                    )
                ).add_to(slp_fg)

# ---- H/L MARKERS ---------------------------------------------------------
hl_fg = folium.FeatureGroup(name='H/L Centers', show=None)
if hl_centers:
    for c in hl_centers:
        color  = 'black'
        shadow = '1px 1px 0 white,-1px -1px 0 white,1px -1px 0 white,-1px 1px 0 white'
        html = (
            f'<div style="display:flex;flex-direction:column;align-items:center;pointer-events:none">'
            f'<div style="font-size:59px;font-weight:900;color:{color};'
            f'font-family:Palatino Linotype,Palatino,serif;line-height:1;text-shadow:{shadow};">{c["type"]}</div>'
            f'</div>'
        )
        folium.Marker(
            location=[c['lat'], c['lon']],
            icon=folium.DivIcon(html=html, icon_size=(60, 44), icon_anchor=(30, 12)),
            tooltip=c['type']
        ).add_to(hl_fg)


# ---- STATION MODELS ------------------------------------------------------
stn_fg = folium.FeatureGroup(name='Station Models', show=True)
visible = list(metar_records)

for d in visible:
    svg_str, sw, sh = station_model_svg(d, S=SYMBOL_SCALE)
    fc_color = flight_cat_color(d)
    popup_html = (
        f'<div style="font-family:monospace;font-size:12px;min-width:200px">'
        f'<b style="font-size:14px;color:#1a4a8a">{d["icao"]}</b> '
        f'<span style="color:{fc_color};font-weight:bold">{d["flt_cat"]}</span><br>'
        f'<span style="color:#888;font-size:10px">{d["name"]}</span><br>'
        f'<hr style="margin:4px 0">'
        f'Temp: <b>{d["temp"]}C</b> &nbsp; Dew: <b>{d["dew"]}C</b><br>'
        f'Wind: <b>{d["wind_dir"]}/{d["wind_spd"]}kt'
        + (f' G{d["wind_gust"]}' if d.get('wind_gust') else '')
        + f'</b><br>'
        f'Vis: <b>{d["vis"]} SM</b> &nbsp; Wx: <b>{d["weather"] or "NIL"}</b><br>'
        f'SLP: <b>{d["slp"]} hPa</b> &nbsp; RH: <b>{d["rh"]}%</b><br>'
        f'Cloud: <b>' + ' '.join(c['raw'] for c in d['clouds']) + '</b><br>'
        f'<a href="https://aviationweather.gov/api/data/metar?ids={d["icao"]}&hours=24&taf=1" '
        f'target="_blank" style="font-size:10px;color:#1a4a8a;">METAR+TAF: {d["icao"]} ↗</a></div>'
    )
    folium.Marker(
        location=[d['lat'], d['lon']],
        icon=folium.DivIcon(html=svg_str, icon_size=(sw, sh),
                            icon_anchor=(sw/2, sh/2), class_name=''),
        popup=folium.Popup(popup_html, max_width=260),
        tooltip=f'{d["icao"]} {d["temp"]}C/{d["dew"]}C SLP={d["slp"]}'
    ).add_to(stn_fg)
# stn_fg.add_to(m)  # disabled: JS dropdown controls station rendering

# ---- BUILD PER-TIMESTAMP STATION DATA for JS dropdown --------------------
import json as _json2
_ts_all = sorted(set(d['timestamp'] for d in metar_records if d['timestamp']))
_ts_data = {}
for _ts in _ts_all:
    _entries = []
    for _d in metar_records:
        if _d['timestamp'] != _ts: continue
        pass  # no geographic filter — use all stations
        _svg, _sw, _sh = station_model_svg(_d, S=SYMBOL_SCALE)
        _fc = flight_cat_color(_d)
        _wg = f' G{_d["wind_gust"]}' if _d.get('wind_gust') else ''
        _pop = (
            f'<div style="font-family:monospace;font-size:12px;min-width:200px">'
            f'<b style="font-size:14px;color:#1a4a8a">{_d["icao"]}</b> '
            f'<span style="color:{_fc};font-weight:bold">{_d["flt_cat"]}</span><br>'
            f'<span style="color:#888;font-size:10px">{_d["name"]}</span>'
            f'<hr style="margin:4px 0">'
            f'Temp/Dew: <b>{_d["temp"]}C / {_d["dew"]}C</b><br>'
            f'Wind: <b>{_d["wind_dir"]}/{_d["wind_spd"]}kt{_wg}</b><br>'
            f'Vis: <b>{_d["vis"]} SM</b> Wx: <b>{_d["weather"] or "NIL"}</b><br>'
            f'SLP: <b>{_d["slp"]} hPa</b> RH: <b>{_d["rh"]}%</b><br>'
            f'Cloud: <b>' + ' '.join(c['raw'] for c in _d['clouds']) + '</b><br>'
            f'<a href="https://aviationweather.gov/api/data/metar?ids={_d["icao"]}&hours=24&taf=1" '
            f'target="_blank" style="font-size:10px;color:#1a4a8a;">METAR+TAF: {_d["icao"]} ↗</a></div>'
        )
        _entries.append({
            'lat': _d['lat'], 'lon': _d['lon'],
            'svg': _svg, 'sw': _sw, 'sh': _sh, 'popup': _pop,
            'tip': f'{_d["icao"]} {_d["temp"]}C/{_d["dew"]}C {_d["wind_dir"]}/{_d["wind_spd"]}kt'
        })
    _ts_data[_ts] = _entries
_ts_json_str = _json2.dumps(_ts_data)
_ts_list_str = _json2.dumps(_ts_all)
_latest_ts   = _ts_all[-1] if _ts_all else ''
print(f'Timestamps available: {_ts_all}')
# ---- end per-timestamp data -----------------------------------------------

folium.LayerControl(collapsed=False).add_to(m)



# ---- TIMESTEP DROPDOWN BAR -----------------------------------------------
ts_bar_html = (
    '<div id="syn-ts-bar" style="'
    'position:fixed;bottom:26px;left:10px;z-index:10000;'
    'background:rgba(255,255,255,0.96);border:1px solid #ccc;border-radius:8px;'
    'padding:6px 12px;font-family:Courier New,monospace;font-size:12px;'
    'box-shadow:0 2px 10px rgba(0,0,0,0.15);display:flex;align-items:center;gap:8px;">'
    '<b style="color:#1a4a8a">Time:</b>'
    '<select id="ts-select" onchange="synUpdateTS(this.value)" '
    'style="font-family:Courier New,monospace;font-size:12px;padding:2px 5px;'
    'border:1px solid #aac;border-radius:4px;background:#f8fbff;color:#1a4a8a;cursor:pointer">'
    '</select>'
    '<span id="ts-count" style="color:#888;font-size:10px;min-width:60px"></span>'
    '<button id="btn-slp" onclick="synToggleLayer(\'slp\')" '
    'style="font-size:9px;padding:2px 7px;cursor:pointer;border:1px solid #aaa;'
    'border-radius:3px;background:#e8f0fe;color:#1a3a6a">Isobars ✓</button>'
    '<button id="btn-hl" onclick="synToggleLayer(\'hl\')" '
    'style="font-size:9px;padding:2px 7px;cursor:pointer;border:1px solid #aaa;'
    'border-radius:3px;background:#e8f0fe;color:#1a3a6a">H/L ✓</button>'
    '<button id="btn-svg" onclick="synToggleLayer(\'svg\')" '
    'style="font-size:9px;padding:2px 7px;cursor:pointer;border:1px solid #aaa;'
    'border-radius:3px;background:#e8f0fe;color:#1a3a6a">Stn ✓</button>'
    '</div>'
)
m.get_root().html.add_child(Element(ts_bar_html))

# ---- TIMESTEP JS --------------------------------------------------------
ts_js = (
    '<script>\n'
    'var _SYN_TS_DATA = ' + _ts_json_str + ';\n'
    'var _SYN_TS_LIST = ' + _ts_list_str + ';\n'
    'var _SYN_SLP     = ' + _ts_slp_json_str + ';\n'
    'var _synStnLayer = null;\n'
    'var _synSlpLayer = null;\n'
    'var _synHLLayer  = null;\n'
    'function synUpdateTS(ts) {\n'
    '  var entries = _SYN_TS_DATA[ts] || [];\n'
    '  var countEl = document.getElementById("ts-count");\n'
    '  if (countEl) countEl.textContent = entries.length + " stns";\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) { console.warn("synUpdateTS: map not ready"); return; }\n'
    '  var MAP = window[keys[0]];\n'
    '  if (!MAP || typeof MAP.removeLayer !== "function") { console.warn("synUpdateTS: invalid map"); return; }\n'
'  if (_synStnLayer) { MAP.removeLayer(_synStnLayer); _synStnLayer = null; }\n'
    '  if (_synSlpLayer) { MAP.removeLayer(_synSlpLayer); _synSlpLayer = null; }\n'
    '  if (_synHLLayer)  { MAP.removeLayer(_synHLLayer);  _synHLLayer  = null; }\n'
    '  _synStnLayer = L.layerGroup();\n'
    '  entries.forEach(function(d) {\n'
    '    L.marker([d.lat, d.lon], {\n'
    '      icon: L.divIcon({\n'
    '        html: d.svg, iconSize:[d.sw,d.sh], iconAnchor:[d.sw/2,d.sh/2], className:""\n'
    '      }), zIndexOffset:500\n'
    '    }).bindPopup(d.popup,{maxWidth:280,closeButton:true}).bindTooltip(d.tip).addTo(_synStnLayer);\n'
    '  });\n'
    '  _synStnLayer.addTo(MAP);\n'
    '  var slpData = _SYN_SLP[ts] || {contours:[], hl:[]};\n'
    '  _synSlpLayer = L.layerGroup();\n'
    '  slpData.contours.forEach(function(ct) {\n'
    '    var latlngs = ct.coords.map(function(c){return [c[1],c[0]];});\n'
    '    L.polyline(latlngs, {\n'
    '      color:"#000000", weight:1, opacity:1.0\n'
    '    }).bindTooltip(Math.round(ct.level)+" ").addTo(_synSlpLayer);\n'
'    if (Math.round(ct.level) % 4 === 0) {\n'
    '      L.marker([ct.label_lat, ct.label_lon], {\n'
    '        icon: L.divIcon({\n'
'          html: \'<div style="font-size:13px;font-weight:normal;color:#000000;\'\n'
    '               +\'font-family:Courier New,monospace;white-space:nowrap;\'\n'
    '               +\'background:rgba(255,255,255,0.9);\'\n'
    '               +\'padding:1px 4px;border-radius:3px;\'\n'
    '               +\'border:1px solid rgba(0,0,0,0.2);">\'\n'
    '               + Math.round(ct.level) + \'</div>\',\n'
    '          iconSize:[42,18], iconAnchor:[21,9], className:""\n'
    '        }),\n'                     # ← comma after divIcon
    '        zIndexOffset: -300\n'      # ← inside marker options
    '      }).addTo(_synSlpLayer);\n'
    '    }\n'
    '  });\n'
    '  (function() {\n'
    '    var REF_LON = -125.0;\n'
    '    var LAT_MIN = 39.0;\n'
    '    var LAT_MAX = 67.0;\n'
    '    var MAX_DLON = 2.5;\n'
    '    var byLevel = {};\n'
    '    slpData.contours.forEach(function(ct) {\n'
    '      var lvl = Math.round(ct.level);\n'
    '      ct.coords.forEach(function(c) {\n'
    '        var dlon = Math.abs(c[0] - REF_LON);\n'
    '        if (dlon > MAX_DLON) return;\n'
    '        if (c[1] < LAT_MIN || c[1] > LAT_MAX) return;\n'
    '        if (c[0] > REF_LON) return;\n'
    '        if (!byLevel[lvl]) byLevel[lvl] = [];\n'
    '        byLevel[lvl].push({lat: c[1], lon: c[0], dlon: dlon});\n'
    '      });\n'
    '    });\n'
    '    Object.keys(byLevel).forEach(function(lvl) {\n'
    '      var candidates = byLevel[lvl];\n'
    '      candidates.sort(function(a, b) { return a.dlon - b.dlon; });\n'
    '      var placed = [];\n'
    '      candidates.forEach(function(c) {\n'
    '        var tooClose = placed.some(function(p) {\n'
    '          return Math.abs(p.lat - c.lat) < 5.0;\n'
    '        });\n'
    '        if (tooClose) return;\n'
    '        placed.push(c);\n'
    '        L.marker([c.lat, REF_LON], {\n'
    '          icon: L.divIcon({\n'
    '            html: \'<div style="font-size:13px;font-weight:normal;color:#000000;\'\n'
    '                 +\'font-family:Courier New,monospace;white-space:nowrap;\'\n'
    '                 +\'background:rgba(255,255,255,0.9);\'\n'
    '                 +\'padding:1px 4px;border-radius:3px;\'\n'
    '                 +\'border:1px solid rgba(0,0,0,0.2);">\'\n'
    '                 + lvl + \'</div>\',\n'
    '            iconSize:[42,18], iconAnchor:[21,9], className:""\n'
    '          }),\n'
    '          zIndexOffset: -300\n'
    '        }).addTo(_synSlpLayer);\n'
    '      });\n'
    '    });\n'
    '  })();\n'
    '  _synHLLayer = L.layerGroup();\n'
    '  slpData.hl.forEach(function(c) {\n'
    '    var color = "black";\n'
    '    var shadow = "1px 1px 0 white,-1px -1px 0 white,1px -1px 0 white,-1px 1px 0 white";\n'
    '    var html = \'<div style="display:flex;flex-direction:column;align-items:center;">\'\n'
    '             + \'<div style="font-size:59px;font-weight:900;color:\'+color+\';\'\n'
    '             + \'font-family:Palatino Linotype,Palatino,serif;line-height:1;text-shadow:\'+shadow+\';">\'+c.type+\'</div>\'\n'
    '             + (c.val !== undefined ? \'<div style="font-size:14px;font-weight:bold;color:\'+color+\';font-family:Courier New,monospace;text-shadow:\'+shadow+\';\">\'+Math.round(c.val)+\'</div>\' : \'\')\n'
    '             + \'</div>\';\n'
    '    L.marker([c.lat, c.lon], {\n'
    '      icon: L.divIcon({html:html, iconSize:[60,44], iconAnchor:[30,12], className:""}),\n'
    '      zIndexOffset: -200\n'
    '    }).bindTooltip(c.type).addTo(_synHLLayer);\n'
    '  });\n'
    '  if (_synShowSlp) _synSlpLayer.addTo(MAP);\n'   # ← ADD THIS
    '  if (_synShowHL)  _synHLLayer.addTo(MAP);\n'    # ← ADD THIS
    '}\n'
    'var _synShowSlp  = true;\n'
    'var _synShowHL   = true;\n'
    'var _synSvgMode  = "colour";\n'
    'function synToggleLayer(which) {\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) return;\n'
    '  var MAP = window[keys[0]];\n'
    '  if (which === "slp") {\n'
    '    _synShowSlp = !_synShowSlp;\n'
    '    var btn = document.getElementById("btn-slp");\n'
    '    if (_synShowSlp) {\n'
    '      if (_synSlpLayer) _synSlpLayer.addTo(MAP);\n'
    '      btn.textContent = "Isobars ✓"; btn.style.background = "#e8f0fe";\n'
    '    } else {\n'
    '      if (_synSlpLayer) MAP.removeLayer(_synSlpLayer);\n'
    '      btn.textContent = "Isobars ✗"; btn.style.background = "#f0f0f0";\n'
    '    }\n'
    '  } else if (which === "hl") {\n'
    '    _synShowHL = !_synShowHL;\n'
    '    var btn2 = document.getElementById("btn-hl");\n'
    '    if (_synShowHL) {\n'
    '      if (_synHLLayer) _synHLLayer.addTo(MAP);\n'
    '      btn2.textContent = "H/L ✓"; btn2.style.background = "#e8f0fe";\n'
    '    } else {\n'
    '      if (_synHLLayer) MAP.removeLayer(_synHLLayer);\n'
    '      btn2.textContent = "H/L ✗"; btn2.style.background = "#f0f0f0";\n'
    '    }\n'
    '  } else if (which === "svg") {\n'
    '    _synSvgMode = (_synSvgMode === "colour") ? "wmo" : "colour";\n'
    '    var btn3 = document.getElementById("btn-svg");\n'
    '    var styleEl = document.getElementById("syn-wx-style");\n'
    '    if (!styleEl) {\n'
    '      styleEl = document.createElement("style");\n'
    '      styleEl.id = "syn-wx-style";\n'
    '      document.head.appendChild(styleEl);\n'
    '    }\n'
    '    if (_synSvgMode === "wmo") {\n'
    '      styleEl.textContent = ".syn-wx-box { display: none !important; }";\n'
    '      btn3.textContent = "WMO ✓"; btn3.style.background = "#f4e8c8"; btn3.style.color = "#5c2e00"; btn3.style.borderColor = "#9a6a00";\n'
    '    } else {\n'
    '      styleEl.textContent = "";\n'
    '      btn3.textContent = "Stn ✓"; btn3.style.background = "#e8f0fe"; btn3.style.color = "#1a3a6a"; btn3.style.borderColor = "#aaa";\n'
    '    }\n'
    '  }\n'
    '}\n'
    'function synTsToUtc(ts) {\n'
    '  var clean=ts.replace(/Z$/i,""); var dd=parseInt(clean.slice(0,2),10);\n'
    '  var hh=parseInt(clean.slice(2,4),10); var mn=parseInt(clean.slice(4,6)||"0",10);\n'
    '  var now=new Date();\n'
    '  var d=new Date(Date.UTC(now.getUTCFullYear(),now.getUTCMonth(),dd,hh,mn,0));\n'
    '  if(dd-now.getUTCDate()>15) d.setUTCMonth(d.getUTCMonth()-1);\n'
    '  return d;\n'
    '}\n'
'function synInitDropdown() {\n'
    '  var sel = document.getElementById("ts-select");\n'
    '  if (!sel) { setTimeout(synInitDropdown, 200); return; }\n'
    '  var nowUtc = new Date();\n'
    '  var latest = null;\n'
    '  _SYN_TS_LIST.forEach(function(ts) {\n'
    '    if (synTsToUtc(ts) > nowUtc) return;\n'
    '    var opt = document.createElement("option");\n'
    '    opt.value = ts; opt.textContent = ts;\n'
    '    sel.appendChild(opt);\n'
    '    latest = ts;\n'
    '  });\n'
    '  if (!latest && _SYN_TS_LIST.length) latest = _SYN_TS_LIST[0];\n'
    '  if (latest) { sel.value = latest; synUpdateTS(latest); }\n'
    '}\n'
    'if (document.readyState==="complete") { setTimeout(synInitDropdown,500); }\n'
    'else { window.addEventListener("load",function(){setTimeout(synInitDropdown,500);}); }\n'
    '</script>\n'
)
m.get_root().html.add_child(Element(ts_js))
# ---- end timestep JS ------------------------------------------------------

# ---- JS: load Natural Earth borders --------------------------------------
borders_js = (
    '<style>\n'
    '.leaflet-container { background: #ffffff !important; }\n'
    '</style>\n'
    '<script>\n'
    '(function() {\n'
    '  function loadBorders() {\n'
    '    var items = [\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_coastline.geojson",\n'
    '       {color:"#444",weight:1.8,opacity:1.0,fill:false}],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_boundary_lines_land.geojson",\n'
    '       {color:"#333",weight:2.0,opacity:1.0,fill:false}],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces_lines.geojson",\n'
    '       {color:"#777",weight:0.9,opacity:0.85,fill:false}],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson",\n'
    '       "ALBERTA"],\n'
    '      ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_lakes.geojson",\n'
    '       {color:"#5588aa",weight:0.8,opacity:0.9,fill:false}]\n'
    '    ];\n'
    '    var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '    if (!keys.length) { setTimeout(loadBorders, 200); return; }\n'
    '    var MAP = window[keys[0]];\n'
    '    items.forEach(function(item) {\n'
    '      if (item[1] === "ALBERTA") {\n'
    '        fetch(item[0]).then(function(r){return r.json();}).then(function(gj){\n'
    '          var ab = {type:"FeatureCollection", features: gj.features.filter(function(f){\n'
    '            var n = (f.properties.name || f.properties.NAME || "").toUpperCase();\n'
    '            return n === "ALBERTA";\n'
    '          })};\n'
    '          L.geoJSON(ab,{style:function(){return {color:"#cc0000",weight:3.5,opacity:1.0,fill:false};}}).addTo(MAP);\n'
    '        }).catch(function(e){console.warn("Alberta border load failed",e);});\n'
    '      } else {\n'
    '        fetch(item[0]).then(function(r){return r.json();}).then(function(gj){\n'
    '          L.geoJSON(gj,{style:function(){return item[1];}}).addTo(MAP);\n'
    '        }).catch(function(e){console.warn("border load failed",e);});\n'
    '      }\n'
    '    });\n'
    '  }\n'
    '  if (document.readyState==="complete") { setTimeout(loadBorders,600); }\n'
    '  else { window.addEventListener("load",function(){setTimeout(loadBorders,600);}); }\n'
    '})();\n'
    '</script>'
)
m.get_root().html.add_child(Element(borders_js))

# ---- SAVE PNG BUTTON -----------------------------------------------------
save_btn_html = (
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>\n'
    '<script>\n'
'function synSavePNG() {\n'
    '  var btn    = document.getElementById("btn-save-png");\n'
    '  var status = document.getElementById("save-status");\n'
    '  btn.disabled = true;\n'
    '  btn.textContent = "Capturing...";\n'
    '  status.textContent = "";\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) { status.textContent="Map not found"; btn.disabled=false; return; }\n'
    '  var MAP   = window[keys[0]];\n'
    '  var mapEl = document.getElementById(keys[0]) || document.querySelector(".leaflet-container");\n'
    '  if (!mapEl) { status.textContent="Map el not found"; btn.disabled=false; return; }\n'
    '  var hideEls = [\n'
    '    mapEl.querySelector(".leaflet-control-container"),\n'
    '    document.querySelector(".leaflet-control-layers"),\n'
    '    document.querySelector(".leaflet-control-zoom"),\n'
    '    document.querySelector(".leaflet-control-attribution"),\n'
    '    document.getElementById("syn-ts-bar"),\n'
    '    document.getElementById("syn-save-bar"),\n'
    '    document.getElementById("syn-fs-btn")\n'
    '  ].filter(Boolean);\n'
    '  var prevVis = hideEls.map(function(el){ return el.style.visibility; });\n'
    '  hideEls.forEach(function(el){ el.style.visibility="hidden"; });\n'
    '  html2canvas(mapEl, {\n'
    '    useCORS: true, allowTaint: true, scale: 2, logging: false\n'
    '  }).then(function(canvas) {\n'
    '    hideEls.forEach(function(el,i){ el.style.visibility=prevVis[i]; });\n'
    '    // convert lat/lon crop bounds to pixel coords using Leaflet\n'
    '    var mapRect = mapEl.getBoundingClientRect();\n'
    '    var scale   = 2;\n'
    '    var tl = MAP.latLngToContainerPoint([65.71423563397606, -134.28652654118332]);\n'
    '    var br = MAP.latLngToContainerPoint([44.51251256829053,  -98.50745737363381]);\n'
    '    var x1 = Math.max(0, tl.x) * scale;\n'
    '    var y1 = Math.max(0, tl.y) * scale;\n'
    '    var x2 = Math.min(mapEl.offsetWidth,  br.x) * scale;\n'
    '    var y2 = Math.min(mapEl.offsetHeight, br.y) * scale;\n'
    '    var cropW = x2 - x1;\n'
    '    var cropH = y2 - y1;\n'
    '    if (cropW <= 0 || cropH <= 0) {\n'
    '      status.textContent = "Crop out of view — pan map first";\n'
    '      btn.disabled=false; btn.textContent="\\uD83D\\uDCF7 Save PNG"; return;\n'
    '    }\n'
    '    var out = document.createElement("canvas");\n'
    '    out.width  = cropW;\n'
    '    out.height = cropH;\n'
    '    out.getContext("2d").drawImage(canvas, x1, y1, cropW, cropH, 0, 0, cropW, cropH);\n'
    '    var ts   = (document.getElementById("ts-select")||{}).value||"synoptic";\n'
    '    var name = "synoptic_" + ts.replace(/[^a-zA-Z0-9]/g,"_") + ".png";\n'
    '    var link = document.createElement("a");\n'
    '    link.download = name;\n'
    '    link.href = out.toDataURL("image/png");\n'
    '    link.click();\n'
    '    btn.disabled=false; btn.textContent="\\uD83D\\uDCF7 Save PNG";\n'
    '    status.textContent="Saved!"; setTimeout(function(){ status.textContent=""; },3000);\n'
    '  }).catch(function(e) {\n'
    '    hideEls.forEach(function(el,i){ el.style.visibility=prevVis[i]; });\n'
    '    status.textContent="Failed: "+e.message;\n'
    '    btn.disabled=false; btn.textContent="\\uD83D\\uDCF7 Save PNG";\n'
    '  });\n'
    '}\n'
    '</script>\n'
)
m.get_root().html.add_child(Element(save_btn_html))

# ---- FULLSCREEN BUTTON ---------------------------------------------------
fullscreen_html = (
    '<style>\n'
    '#syn-fs-btn {\n'
    '  position:fixed;top:10px;left:10px;z-index:10001;\n'
    '  background:rgba(255,255,255,0.96);border:1px solid #aaa;border-radius:6px;\n'
    '  padding:5px 10px;font-family:Courier New,monospace;font-size:12px;\n'
    '  box-shadow:0 2px 8px rgba(0,0,0,0.15);cursor:pointer;color:#1a3a6a;\n'
    '}\n'
    '#syn-fs-btn:hover { background:#e8f0fe; }\n'
    '.syn-fs-active {\n'
    '  position:fixed!important;top:0!important;left:0!important;\n'
    '  width:100vw!important;height:100vh!important;\n'
    '  z-index:9999!important;margin:0!important;\n'
    '}\n'
    '</style>\n'
    '<button id="syn-fs-btn" onclick="synToggleFS()">&#x26F6; Fullscreen</button>\n'
    '<script>\n'
    'var _synFS = false;\n'
    'var _synMapEl = null;\n'
    'var _synOrigStyle = "";\n'
    'function synToggleFS() {\n'
    '  var btn = document.getElementById("syn-fs-btn");\n'
    '  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});\n'
    '  if (!keys.length) { console.warn("map not found"); return; }\n'
    '  var MAP = window[keys[0]];\n'
    '  if (!_synMapEl) {\n'
    '    _synMapEl = document.getElementById(keys[0]);\n'
    '    if (!_synMapEl) _synMapEl = document.querySelector(".leaflet-container");\n'
    '  }\n'
    '  if (!_synMapEl) { console.warn("map element not found"); return; }\n'
    '  _synFS = !_synFS;\n'
    '  if (_synFS) {\n'
    '    _synOrigStyle = _synMapEl.getAttribute("style") || "";\n'
    '    _synMapEl.setAttribute("style",\n'
    '      "position:fixed!important;top:0;left:0;"\n'
    '      +"width:100vw!important;height:100vh!important;"\n'
    '      +"z-index:9999!important;margin:0!important;");\n'
    '    btn.innerHTML = "&#x274C; Exit Fullscreen";\n'
    '  } else {\n'
    '    _synMapEl.setAttribute("style", _synOrigStyle);\n'
    '    btn.innerHTML = "&#x26F6; Fullscreen";\n'
    '  }\n'
    '  setTimeout(function(){ MAP.invalidateSize(); }, 100);\n'
    '}\n'
    '</script>\n'
)
m.get_root().html.add_child(Element(fullscreen_html))
m.get_root().html.add_child(Element(fire_zones_html))

# save & display
out_path = 'output/synoptic_map.html'
m.save(out_path)
print(f'Map saved: {out_path}')
print(f'Stations: {len(visible)}')

with open(out_path) as _f:
    _html = _f.read()
# Last lines of Cell 9 — fix the display call











# ── Cell 11 ────────────────────────────────────────────────────────────────
from datetime import datetime, timezone
_utc_hour = datetime.now(timezone.utc).hour
if   _utc_hour <  6: EXPORT_TIME = "0000Z"
elif _utc_hour < 12: EXPORT_TIME = "0600Z"
elif _utc_hour < 18: EXPORT_TIME = "1200Z"
else:                EXPORT_TIME = "1800Z"
print(f'UTC hour: {_utc_hour}  →  default export: {EXPORT_TIME}')
with open('output/synoptic_map.html', 'r', encoding='utf-8') as f:


    html = f.read()

new_fn = '''function synSavePNG() {
  var btn    = document.getElementById("btn-save-png");
  var status = document.getElementById("save-status");
  if (btn) { btn.disabled = true; btn.textContent = "Capturing..."; }
  if (status) status.textContent = "";

  var keys = Object.keys(window).filter(function(k){ return k.startsWith("map_"); });
  if (!keys.length) { if(status) status.textContent="Map not found"; if(btn) btn.disabled=false; return; }
  var MAP   = window[keys[0]];
  var mapEl = document.getElementById(keys[0]) || document.querySelector(".leaflet-container");
  if (!mapEl) { if(status) status.textContent="Map el not found"; if(btn) btn.disabled=false; return; }

  var hideEls = [
    mapEl.querySelector(".leaflet-control-container"),
    document.querySelector(".leaflet-control-layers"),
    document.querySelector(".leaflet-control-zoom"),
    document.querySelector(".leaflet-control-attribution"),
    document.getElementById("syn-ts-bar"),
    document.getElementById("syn-save-bar"),
    document.getElementById("syn-fs-btn")
  ].filter(Boolean);
  var prevVis = hideEls.map(function(el){ return el.style.visibility; });
  hideEls.forEach(function(el){ el.style.visibility = "hidden"; });

  var CENTER = [55, -102];
  var ZOOM   = 5;
  var TARGET_W = 1400;
  var TARGET_H = 1100;

  var origW = mapEl.style.width;
  var origH = mapEl.style.height;

  function restore() {
    mapEl.style.width  = origW;
    mapEl.style.height = origH;
    MAP.invalidateSize();
    if (btn) { btn.disabled = false; btn.textContent = "Save PNG"; }
  }

  mapEl.style.width  = TARGET_W + "px";
  mapEl.style.height = TARGET_H + "px";
  MAP.invalidateSize();

  setTimeout(function() {
    MAP.setView(CENTER, ZOOM, { animate: false });
    setTimeout(function() {
      html2canvas(mapEl, {
        useCORS: true, allowTaint: true,
        scale: 2, logging: false,
        width: TARGET_W, height: TARGET_H
      }).then(function(canvas) {
        hideEls.forEach(function(el, i){ el.style.visibility = prevVis[i]; });

        var cropH = canvas.height;
        var cropW = Math.min(Math.round(cropH * 8.5 / 11.0), canvas.width);
        var out = document.createElement("canvas");
        out.width  = cropW;
        out.height = cropH;
        var ctx2 = out.getContext("2d");
        ctx2.drawImage(canvas, 0, 0, cropW, cropH, 0, 0, cropW, cropH);

        // ── White out outside frame ────────────────────────────────────
        var MARGIN = 36;
        ctx2.fillStyle = "rgba(255,255,255,1.0)";
        ctx2.fillRect(0,              0,              cropW,  MARGIN);        // top
        ctx2.fillRect(0,              cropH - MARGIN, cropW,  MARGIN);        // bottom
        ctx2.fillRect(0,              0,              MARGIN, cropH);         // left
        ctx2.fillRect(cropW - MARGIN, 0,              MARGIN, cropH);         // right
        // ──────────────────────────────────────────────────────────────

        // ── Timestamp label box (bottom-left) ──────────────────────────
        var today  = new Date();
        var months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];
        var dows   = ["SUNDAY","MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY"];
        var dowStr  = dows[today.getUTCDay()];
        var dateStr = months[today.getUTCMonth()] + " " + String(today.getUTCDate()).padStart(2,"0") + " " + today.getUTCFullYear();
        var selEl   = document.getElementById("ts-select");
        var tsVal   = selEl ? selEl.value : "";
        var timeStr = tsVal ? tsVal.slice(2) : "1200Z";
        var lines   = ["SURFACE MAP", dowStr + " " + dateStr, timeStr];
        var fSize   = 36;
        var pad     = 24;
        var lineH   = fSize * 1.3;
        var boxH    = lines.length * lineH + pad * 2;
        ctx2.font   = fSize + "px Arial, sans-serif";
        var maxW    = Math.max.apply(null, lines.map(function(l){ return ctx2.measureText(l).width; }));
        var boxW    = maxW + pad * 2;
        var bx      = MARGIN;
        var by      = cropH - MARGIN - boxH;
        ctx2.fillStyle = "rgba(255,255,255,0.88)";
        ctx2.fillRect(bx, by, boxW, boxH);
        ctx2.strokeStyle = "#1a4a8a";
        ctx2.lineWidth = 3;
        ctx2.strokeRect(bx, by, boxW, boxH);
        ctx2.fillStyle    = "#1a2030";
        ctx2.textBaseline = "top";
        ctx2.textAlign    = "center";
        var centerX = bx + boxW / 2;
        lines.forEach(function(line, i) {
          ctx2.font = fSize + "px Arial, sans-serif";
          ctx2.fillText(line, centerX, by + pad + i * lineH);
        });
// ── Frame border only ─────────────────────────────────────────
        var MARGIN = 36;
        ctx2.strokeStyle = "#1a2030";
        ctx2.lineWidth   = 2;
        ctx2.strokeRect(MARGIN, MARGIN, cropW - MARGIN * 2, cropH - MARGIN * 2);
        // ──────────────────────────────────────────────────────────────

// ── Corner lat/lon labels ─────────────────────────────────────
        var SC   = 2;
        var tlLL = MAP.containerPointToLatLng([MARGIN/SC,            MARGIN/SC]);
        var trLL = MAP.containerPointToLatLng([TARGET_W - MARGIN/SC, MARGIN/SC]);
        var blLL = MAP.containerPointToLatLng([MARGIN/SC,            TARGET_H - MARGIN/SC]);
        var brLL = MAP.containerPointToLatLng([TARGET_W - MARGIN/SC, TARGET_H - MARGIN/SC]);
        function fmtLat(v){ return Math.abs(v).toFixed(1)+(v>=0?"°N":"°S"); }
        function fmtLon(v){ return Math.abs(v).toFixed(1)+(v>=0?"°E":"°W"); }
        ctx2.font         = "18px Arial, sans-serif";
        ctx2.fillStyle    = "#1a2030";
        ctx2.textBaseline = "middle";
        var LAT_PAD = 30;
        [{ll:tlLL,x:MARGIN/2,y:MARGIN+LAT_PAD,r:-Math.PI/2},
         {ll:blLL,x:MARGIN/2,y:cropH-MARGIN-LAT_PAD,r:-Math.PI/2},
         {ll:trLL,x:cropW-MARGIN/2,y:MARGIN+LAT_PAD,r:Math.PI/2},
         {ll:brLL,x:cropW-MARGIN/2,y:cropH-MARGIN-LAT_PAD,r:Math.PI/2}
        ].forEach(function(p){
          ctx2.save(); ctx2.translate(p.x,p.y); ctx2.rotate(p.r);
          ctx2.textAlign="center"; ctx2.fillText(fmtLat(p.ll.lat),0,0); ctx2.restore();
        });
        var LON_PAD = 15;
        ctx2.textAlign="left";
        ctx2.fillText(fmtLon(tlLL.lng), MARGIN+LON_PAD,            MARGIN/2);
        ctx2.fillText(fmtLon(blLL.lng), MARGIN+LON_PAD,            cropH-MARGIN/2);
        ctx2.textAlign="right";
        ctx2.fillText(fmtLon(trLL.lng), cropW-MARGIN-LON_PAD,      MARGIN/2);
        ctx2.fillText(fmtLon(brLL.lng), cropW-MARGIN-LON_PAD,      cropH-MARGIN/2);
        // ──────────────────────────────────────────────────────────────



        // ── Export timestamp (bottom-right) ───────────────────────────
        var expNow = new Date();
        var expStr = "Exported at: "
          + expNow.getUTCFullYear() + "/"
          + String(expNow.getUTCMonth()+1).padStart(2,"0") + "/"
          + String(expNow.getUTCDate()).padStart(2,"0") + " "
          + String(expNow.getUTCHours()).padStart(2,"0") + ":"
          + String(expNow.getUTCMinutes()).padStart(2,"0") + ":"
          + String(expNow.getUTCSeconds()).padStart(2,"0") + "Z";
        ctx2.font         = "8px Arial, sans-serif";
        ctx2.fillStyle    = "#555555";
        ctx2.textBaseline = "middle";
        ctx2.textAlign    = "right";
        var lonLabelWidth = ctx2.measureText(fmtLon(brLL.lng)).width;
        ctx2.fillText(expStr, cropW - MARGIN - LON_PAD - lonLabelWidth - 60, cropH - MARGIN/2);
        // ──────────────────────────────────────────────────────────────


        // ── Build filename surface_plot_YYYYMMDDHHZ.png ─────────────────────
        var now = new Date();
        var yyyy = now.getUTCFullYear();
        var mm   = String(now.getUTCMonth()+1).padStart(2,"0");
        var dd   = String(now.getUTCDate()).padStart(2,"0");
        var selEl2  = document.getElementById("ts-select");
        var tsVal2  = selEl2 ? selEl2.value : "";
        // Extract HH from the timestamp value shown in the bottom-left box (e.g. "2024010112Z" → "12")
        // tsVal2 format: "131200Z" → strip Z, take chars at position 2-3 = "12"
        var tsStripped = tsVal2.replace(/Z$/i, "");
        var hh = tsStripped.length >= 4 ? tsStripped.slice(-4, -2) : "12";

        var name = "surface_plot_" + yyyy + mm + dd + hh + "Z-" + (window._synMetarPNG ? "no_contour" : "with_contour") + ".png";

        var link = document.createElement("a");
        link.download = name;
        link.href = out.toDataURL("image/png");
        link.click();
        restore();
        if (status) { status.textContent = "Saved!"; setTimeout(function(){ status.textContent = ""; }, 3000); }

      }).catch(function(e) {
        hideEls.forEach(function(el, i){ el.style.visibility = prevVis[i]; });
        restore();
        if (status) status.textContent = "Failed: " + e.message;
      });
    }, 300);
  }, 200);
}'''

# ── Replace synSavePNG by brace matching ──────────────────────────────
new_fn = new_fn.replace('"{EXPORT_TIME}"', f'"{EXPORT_TIME}"')
start = html.find('function synSavePNG() {')
if start == -1:
    print('ERROR: synSavePNG not found — run Cell 9 first')
else:
    depth = 0
    i = start
    while i < len(html):
        if html[i] == '{': depth += 1
        elif html[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1
    html = html[:start] + new_fn + html[end:]
    print('synSavePNG replaced')

# ── Hide contours and H/L ─────────────────────────────────────────────
html = html.replace('var _synShowSlp = true;', 'var _synShowSlp = false;')
html = html.replace('var _synShowHL  = true;', 'var _synShowHL  = false;')

# ── Inject synExport1200Z ─────────────────────────────────────────────
import re
_body_idx = html.rfind('</body>')
_inject_idx = html.rfind('<script>', 0, _body_idx)
while _inject_idx != -1 and 'synExport1200Z' not in html[_inject_idx:_body_idx]:
    _inject_idx = html.rfind('<script>', 0, _inject_idx)
if _inject_idx != -1 and 'synExport1200Z' in html[_inject_idx:_body_idx]:
    html = html[:_inject_idx] + html[_body_idx:]
if True:  # always inject

    _btn_bg  = "#ffd8a8" if EXPORT_TIME in ("1800Z","0000Z") else "#c8dff4"
    _btn_bdr = "#a85c00" if EXPORT_TIME in ("1800Z","0000Z") else "#1a4a8a"
    _btn_clr = "#5c2e00" if EXPORT_TIME in ("1800Z","0000Z") else "#1a3a6a"
    export_js = '''<script>
function synExport1200Z() {
  var sel = document.getElementById("ts-select");
  if (sel) {
    var target = "{EXPORT_TIME}";
    var opts = Array.from(sel.options).filter(function(o){ return o.value.indexOf(target) !== -1; });
    if (!opts.length) {
      var hh = target.replace("Z","");
      opts = Array.from(sel.options).filter(function(o){ return o.value.slice(-4,-2) === hh || o.value.slice(2,4) === hh; });
    }
    if (opts.length) { sel.value = opts[opts.length-1].value; synUpdateTS(sel.value); }
  }
  setTimeout(synSavePNG, 800);
}
function synExportCurrent() {
  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});
  var MAP = keys.length ? window[keys[0]] : null;
  if (!_synShowSlp && MAP) { _synShowSlp = true; _synSlpLayer.addTo(MAP); var btn=document.getElementById("btn-slp"); if(btn){btn.textContent="Isobars ✓";btn.style.background="#e8f0fe";} }
  if (!_synShowHL  && MAP) { _synShowHL  = true; _synHLLayer.addTo(MAP);  var btn2=document.getElementById("btn-hl");  if(btn2){btn2.textContent="H/L ✓";btn2.style.background="#e8f0fe";} }
  var styleEl = document.getElementById("syn-wx-style");
  if (styleEl) styleEl.textContent = "";
  if (_synSvgMode === "wmo") { _synSvgMode = "colour"; var btn3=document.getElementById("btn-svg"); if(btn3){btn3.textContent="Stn ✓";btn3.style.background="#e8f0fe";btn3.style.color="#1a3a6a";btn3.style.borderColor="#aaa";} var _sel=document.getElementById("ts-select"); if(_sel&&_sel.value) synUpdateTS(_sel.value); }
  setTimeout(synSavePNG, 400);
}
function synExportCurrentMetar() {
  var hadSlp = _synShowSlp, hadHL = _synShowHL;
  if (hadSlp) { _synShowSlp = false; _synSlpLayer.remove(); }
  if (hadHL)  { _synShowHL  = false; _synHLLayer.remove();  }
  var styleEl = document.getElementById("syn-wx-style");
  if (!styleEl) { styleEl = document.createElement("style"); styleEl.id = "syn-wx-style"; document.head.appendChild(styleEl); }
  var hadWxHidden = styleEl.textContent.indexOf("syn-wx-box") !== -1;
  styleEl.textContent = ".syn-wx-box { display: none !important; }";
  window._synMetarPNG = true;
  setTimeout(function() {
    synSavePNG();
    setTimeout(function() {
      if (hadSlp) { _synShowSlp = true; _synSlpLayer.addTo(MAP); }
      if (hadHL)  { _synShowHL  = true; _synHLLayer.addTo(MAP);  }
      if (!hadWxHidden) styleEl.textContent = "";
      window._synMetarPNG = false;
    }, 3000);
  }, 200);
}
function synToggleSvgForMetar(hide, keys) {
  var MAP = window[keys[0]];
  if (hide) {
    if (_synStnLayer) MAP.removeLayer(_synStnLayer);
  } else {
    if (_synShowSvg && _synStnLayer) _synStnLayer.addTo(MAP);
  }
}
function synExportMetar() {
  var hadSlp = _synShowSlp, hadHL = _synShowHL;
  if (hadSlp) { _synShowSlp = false; _synSlpLayer.remove(); }
  if (hadHL)  { _synShowHL  = false; _synHLLayer.remove();  }
  var styleEl = document.getElementById("syn-wx-style");
  if (!styleEl) { styleEl = document.createElement("style"); styleEl.id = "syn-wx-style"; document.head.appendChild(styleEl); }
  var hadWxHidden = styleEl.textContent.indexOf("syn-wx-box") !== -1;
  styleEl.textContent = ".syn-wx-box { display: none !important; }";
  window._synMetarPNG = true;
  setTimeout(function() {
    synSavePNG();
    setTimeout(function() {
      if (hadSlp) { _synShowSlp = true; _synSlpLayer.addTo(MAP); }
      if (hadHL)  { _synShowHL  = true; _synHLLayer.addTo(MAP);  }
      if (!hadWxHidden) styleEl.textContent = "";
      window._synMetarPNG = false;
    }, 3000);
  }, 200);
}
</script>
<div style="position:fixed;top:10px;right:10px;z-index:10002;display:flex;flex-direction:column;gap:6px;">
  <button onclick="synExport1200Z()" style="font-family:Courier New,monospace;font-size:12px;
    padding:5px 12px;background:{BTN_BG};border:1px solid {BTN_BDR};border-radius:5px;
    color:{BTN_CLR};cursor:pointer;font-weight:bold;">&#9928; Export {EXPORT_TIME} Analysis PNG</button>
  <button onclick="synExportMetar()" style="font-family:Courier New,monospace;font-size:12px;
    padding:5px 12px;background:#f4e8c8;border:1px solid #9a6a00;border-radius:5px;
    color:#4a3000;cursor:pointer;font-weight:bold;">&#128225; Export {EXPORT_TIME} METAR PNG</button>
  <button onclick="synExportCurrent()" style="font-family:Courier New,monospace;font-size:12px;
    padding:5px 12px;background:#d4f4c8;border:1px solid #1a6a2a;border-radius:5px;
    color:#1a3a1a;cursor:pointer;font-weight:bold;">&#9200; Export Current Timestep PNG</button>
  <button onclick="synExportCurrentMetar()" style="font-family:Courier New,monospace;font-size:12px;
    padding:5px 12px;background:#e8f4e8;border:1px solid #2a7a3a;border-radius:5px;
    color:#1a4a1a;cursor:pointer;font-weight:bold;">&#128225; Export Current Timestep METAR PNG</button>
  <button onclick="synShowRunPanel()" style="font-family:Courier New,monospace;font-size:12px;
    padding:5px 12px;background:#f0e8f8;border:1px solid #6a2a9a;border-radius:5px;
    color:#3a006a;cursor:pointer;font-weight:bold;"><span id="gha-run-btn-text">&#9881; Run Script Now</span><span id="gha-last-run" style="font-size:9px;font-weight:normal;color:#9a6acc;margin-left:6px;"></span></button>
  <div id="gha-panel" style="display:none;flex-direction:row;align-items:center;gap:6px;padding:5px 10px;
    background:#faf8ff;border:1px solid #9a6acc;border-radius:5px;">
    <span style="color:#555;font-size:11px;font-family:Courier New,monospace;">PIN</span>
    <input id="gha-pin" type="password" maxlength="4" placeholder="····"
      onkeydown="if(event.key==='Enter')synTriggerGHA()"
      style="width:52px;font-family:Courier New,monospace;font-size:12px;padding:3px 5px;
      border:1px solid #9a6acc;border-radius:3px;text-align:center;"/>
    <button onclick="synTriggerGHA()" style="padding:3px 10px;background:#7a2acc;border:none;
      border-radius:3px;color:white;cursor:pointer;font-family:Courier New,monospace;font-size:11px;font-weight:bold;">&#9889; Run</button>
    <span id="gha-status" style="color:#555;font-size:10px;font-family:Courier New,monospace;"></span>
  </div>
  <div id="gha-progress" style="display:none;flex-direction:column;gap:3px;padding:6px 10px;
    background:#faf8ff;border:1px solid #9a6acc;border-radius:5px;">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <b style="color:#3a006a;font-family:Courier New,monospace;font-size:11px;">&#128640; Workflow Progress</b>
      <span id="gha-run-status" style="color:#888;font-size:10px;font-family:Courier New,monospace;"></span>
    </div>
    <div style="background:#e8e0f0;border-radius:3px;height:6px;overflow:hidden;">
      <div id="gha-bar" style="height:100%;width:0%;background:linear-gradient(90deg,#7a2acc,#a855f7);border-radius:3px;transition:width 0.6s ease;"></div>
    </div>
    <div id="gha-steps" style="display:flex;flex-direction:column;gap:2px;font-family:Courier New,monospace;font-size:10px;"></div>
  </div>
</div>'''.replace('{BTN_BG}', _btn_bg).replace('{BTN_BDR}', _btn_bdr).replace('{BTN_CLR}', _btn_clr)



    export_js = export_js.replace('"{EXPORT_TIME}"', f'"{EXPORT_TIME}"')
    export_js = export_js.replace('{EXPORT_TIME}', EXPORT_TIME)
    html = html.replace('</body>', export_js + '</body>')
    print('Export 1200Z injected')
else:
    print('Export 1200Z already present')

# With Contour
# ── Cell 11.1 ────────────────────────────────────────────────────────────────
from datetime import datetime, timezone
_utc_hour = datetime.now(timezone.utc).hour
if   _utc_hour <  6: EXPORT_TIME = "0000Z"
elif _utc_hour < 12: EXPORT_TIME = "0600Z"
elif _utc_hour < 18: EXPORT_TIME = "1200Z"
else:                EXPORT_TIME = "1800Z"
print(f'UTC hour: {_utc_hour}  →  default export: {EXPORT_TIME}')
with open('output/synoptic_map.html', 'r', encoding='utf-8') as f:
    html = f.read()

new_fn = '''function synSavePNG() {
  var btn    = document.getElementById("btn-save-png");
  var status = document.getElementById("save-status");
  if (btn) { btn.disabled = true; btn.textContent = "Capturing..."; }
  if (status) status.textContent = "";

  var keys = Object.keys(window).filter(function(k){ return k.startsWith("map_"); });
  if (!keys.length) { if(status) status.textContent="Map not found"; if(btn) btn.disabled=false; return; }
  var MAP   = window[keys[0]];
  var mapEl = document.getElementById(keys[0]) || document.querySelector(".leaflet-container");
  if (!mapEl) { if(status) status.textContent="Map el not found"; if(btn) btn.disabled=false; return; }

  var hideEls = [
    mapEl.querySelector(".leaflet-control-container"),
    document.querySelector(".leaflet-control-layers"),
    document.querySelector(".leaflet-control-zoom"),
    document.querySelector(".leaflet-control-attribution"),
    document.getElementById("syn-ts-bar"),
    document.getElementById("syn-save-bar"),
    document.getElementById("syn-fs-btn")
  ].filter(Boolean);
  var prevVis = hideEls.map(function(el){ return el.style.visibility; });
  hideEls.forEach(function(el){ el.style.visibility = "hidden"; });

  // ── CENTER + ZOOM: edit here to reframe ────────────────────────────────
  var CENTER = [55, -104];
  var ZOOM   = 5;
  // ───────────────────────────────────────────────────────────────────────

  var TARGET_W = 1400;
  var TARGET_H = 1100;

  var origW = mapEl.style.width;
  var origH = mapEl.style.height;

  function restore() {
    mapEl.style.width  = origW;
    mapEl.style.height = origH;
    MAP.invalidateSize();
    if (btn) { btn.disabled = false; btn.textContent = "Save PNG"; }
  }

  // Step 1: resize
  mapEl.style.width  = TARGET_W + "px";
  mapEl.style.height = TARGET_H + "px";
  MAP.invalidateSize();

  setTimeout(function() {
    // Step 2: set view
    MAP.setView(CENTER, ZOOM, { animate: false });

    setTimeout(function() {
      // Step 3: capture full map
      html2canvas(mapEl, {
        useCORS: true, allowTaint: true,
        scale: 2, logging: false,
        width: TARGET_W, height: TARGET_H
      }).then(function(canvas) {
        hideEls.forEach(function(el, i){ el.style.visibility = prevVis[i]; });

        // Crop legal portrait from top-left: width = height * (8.5/14)
        var cropH = canvas.height;
        var cropW = Math.min(Math.round(cropH * 8.5 / 11.0), canvas.width);


        var out = document.createElement("canvas");
        out.width  = cropW;
        out.height = cropH;
        out.getContext("2d").drawImage(canvas, 0, 0, cropW, cropH, 0, 0, cropW, cropH);

// ── Build filename surface_plot_YYYYMMDDHHZ.png ─────────────────────
        var now = new Date();
        var yyyy = now.getUTCFullYear();
        var mm   = String(now.getUTCMonth()+1).padStart(2,"0");
        var dd   = String(now.getUTCDate()).padStart(2,"0");
        var selElF  = document.getElementById("ts-select");
        var tsValF  = selElF ? selElF.value : "";
        var tsStripped = tsValF.replace(/Z$/i, "");
        var hh = tsStripped.length >= 4 ? tsStripped.slice(-4, -2) : "12";

        var name = "surface_plot_" + yyyy + mm + dd + hh + "Z-" + (window._synMetarPNG ? "no_contour" : "with_contour") + ".png";



// ── Bottom-left stamp ──────────────────────────────────────────
        var ctx2   = out.getContext("2d");
        var today  = new Date();
        var months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];
        var dows   = ["SUNDAY","MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY"];
        var dowStr  = dows[today.getUTCDay()];
        var dateStr = months[today.getUTCMonth()] + " " + String(today.getUTCDate()).padStart(2,"0") + " " + today.getUTCFullYear();
        var selEl   = document.getElementById("ts-select");
        var tsVal   = selEl ? selEl.value : "";
        var timeStr = tsVal ? tsVal.slice(2) : "1200Z";
        var lines   = ["SURFACE MAP", dowStr + " " + dateStr, timeStr];
        var fSize   = 36;
        var pad     = 24;
        var lineH   = fSize * 1.3;
        var boxH    = lines.length * lineH + pad * 2;
        ctx2.font   = fSize + "px Arial, sans-serif";
        var maxW    = Math.max(...lines.map(l => ctx2.measureText(l).width));
        var boxW    = maxW + pad * 2;
        var bx      = 20;
        var by      = out.height - boxH - 20;
        // background
        ctx2.fillStyle = "rgba(255,255,255,0.88)";
        ctx2.fillRect(bx, by, boxW, boxH);
        ctx2.strokeStyle = "#1a4a8a";
        ctx2.lineWidth = 3;
        ctx2.strokeRect(bx, by, boxW, boxH);
        // text — centered
        ctx2.fillStyle    = "#1a2030";
        ctx2.textBaseline = "top";
        ctx2.textAlign    = "center";
        var centerX = bx + boxW / 2;
        lines.forEach(function(line, i) {
          ctx2.font = fSize + "px Arial, sans-serif";
          ctx2.fillText(line, centerX, by + pad + i * lineH);
        });
        ctx2.textAlign = "left";
        // ──────────────────────────────────────────────────────────────

        // ── White out outside frame ────────────────────────────────────
        var MARGIN = 36;
        ctx2.fillStyle = "rgba(255,255,255,1.0)";
        ctx2.fillRect(0,              0,              cropW,  MARGIN);
        ctx2.fillRect(0,              cropH - MARGIN, cropW,  MARGIN);
        ctx2.fillRect(0,              0,              MARGIN, cropH);
        ctx2.fillRect(cropW - MARGIN, 0,              MARGIN, cropH);
        // ── Frame border ──────────────────────────────────────────────
        ctx2.strokeStyle = "#1a2030";
        ctx2.lineWidth   = 2;
        ctx2.strokeRect(MARGIN, MARGIN, cropW - MARGIN * 2, cropH - MARGIN * 2);
        // ──────────────────────────────────────────────────────────────

        // ── Corner lat/lon labels ─────────────────────────────────────
        var SC   = 2;
        var tlLL = MAP.containerPointToLatLng([MARGIN/SC,            MARGIN/SC]);
        var trLL = MAP.containerPointToLatLng([TARGET_W - MARGIN/SC, MARGIN/SC]);
        var blLL = MAP.containerPointToLatLng([MARGIN/SC,            TARGET_H - MARGIN/SC]);
        var brLL = MAP.containerPointToLatLng([TARGET_W - MARGIN/SC, TARGET_H - MARGIN/SC]);
        function fmtLat(v){ return Math.abs(v).toFixed(1)+(v>=0?"°N":"°S"); }
        function fmtLon(v){ return Math.abs(v).toFixed(1)+(v>=0?"°E":"°W"); }
        ctx2.font         = "18px Arial, sans-serif";
        ctx2.fillStyle    = "#1a2030";
        ctx2.textBaseline = "middle";
        var LAT_PAD = 30;
        [{ll:tlLL,x:MARGIN/2,y:MARGIN+LAT_PAD,r:-Math.PI/2},
         {ll:blLL,x:MARGIN/2,y:cropH-MARGIN-LAT_PAD,r:-Math.PI/2},
         {ll:trLL,x:cropW-MARGIN/2,y:MARGIN+LAT_PAD,r:Math.PI/2},
         {ll:brLL,x:cropW-MARGIN/2,y:cropH-MARGIN-LAT_PAD,r:Math.PI/2}
        ].forEach(function(p){
          ctx2.save(); ctx2.translate(p.x,p.y); ctx2.rotate(p.r);
          ctx2.textAlign="center"; ctx2.fillText(fmtLat(p.ll.lat),0,0); ctx2.restore();
        });
        var LON_PAD = 15;
        ctx2.textAlign="left";
        ctx2.fillText(fmtLon(tlLL.lng), MARGIN+LON_PAD,       MARGIN/2);
        ctx2.fillText(fmtLon(blLL.lng), MARGIN+LON_PAD,       cropH-MARGIN/2);
        ctx2.textAlign="right";
        ctx2.fillText(fmtLon(trLL.lng), cropW-MARGIN-LON_PAD, MARGIN/2);
        ctx2.fillText(fmtLon(brLL.lng), cropW-MARGIN-LON_PAD, cropH-MARGIN/2);
        // ──────────────────────────────────────────────────────────────




        // ── Export timestamp (bottom-right) ───────────────────────────
        var expNow = new Date();
        var expStr = "Exported at: "
          + expNow.getUTCFullYear() + "/"
          + String(expNow.getUTCMonth()+1).padStart(2,"0") + "/"
          + String(expNow.getUTCDate()).padStart(2,"0") + " "
          + String(expNow.getUTCHours()).padStart(2,"0") + ":"
          + String(expNow.getUTCMinutes()).padStart(2,"0") + ":"
          + String(expNow.getUTCSeconds()).padStart(2,"0") + "Z";
        ctx2.font         = "8px Arial, sans-serif";
        ctx2.fillStyle    = "#555555";
        ctx2.textBaseline = "middle";
        ctx2.textAlign    = "right";
        var lonLabelWidth = ctx2.measureText(fmtLon(brLL.lng)).width;
        ctx2.fillText(expStr, cropW - MARGIN - LON_PAD - lonLabelWidth - 60, cropH - MARGIN/2);
        // ──────────────────────────────────────────────────────────────



        var link = document.createElement("a");
        link.download = name;
        link.href = out.toDataURL("image/png");
        link.click();

        restore();
        if (status) { status.textContent = "Saved!"; setTimeout(function(){ status.textContent = ""; }, 3000); }

      }).catch(function(e) {
        hideEls.forEach(function(el, i){ el.style.visibility = prevVis[i]; });
        restore();
        if (status) status.textContent = "Failed: " + e.message;
      });

    }, 300); // settle after setView
  }, 200);   // settle after resize
}'''

# Replace entire synSavePNG by brace matching
new_fn = new_fn.replace('"{EXPORT_TIME}"', f'"{EXPORT_TIME}"')
# Replace entire synSavePNG by brace matching
start = html.find('function synSavePNG() {')
if start == -1:
    print('ERROR: synSavePNG not found — run Cell 9 first')
else:
    depth = 0
    i = start
    while i < len(html):
        if html[i] == '{': depth += 1
        elif html[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1
    html = html[:start] + new_fn + html[end:]
    print('synSavePNG replaced')

# ── Ensure contours and H/L are visible ───────────────────────────────
html = html.replace('var _synShowSlp = false;', 'var _synShowSlp = true;')
html = html.replace('var _synShowHL  = false;', 'var _synShowHL  = true;')

# Always re-inject — strip previous injection block completely
_body_idx = html.rfind('</body>')
_inject_idx = html.rfind('<script>', 0, _body_idx)
while _inject_idx != -1 and 'synExport1200Z' not in html[_inject_idx:_body_idx]:
    _inject_idx = html.rfind('<script>', 0, _inject_idx)
if _inject_idx != -1 and 'synExport1200Z' in html[_inject_idx:_body_idx]:
    html = html[:_inject_idx] + html[_body_idx:]
if True:  # always inject
    _btn_bg  = "#ffd8a8" if EXPORT_TIME in ("1800Z","0000Z") else "#c8dff4"
    _btn_bdr = "#a85c00" if EXPORT_TIME in ("1800Z","0000Z") else "#1a4a8a"
    _btn_clr = "#5c2e00" if EXPORT_TIME in ("1800Z","0000Z") else "#1a3a6a"
    export_js = '''<script>
function synExport1200Z() {
  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});
  var MAP = keys.length ? window[keys[0]] : null;
  if (!_synShowSlp && MAP) { _synShowSlp = true; _synSlpLayer.addTo(MAP); var btn=document.getElementById("btn-slp"); if(btn){btn.textContent="Isobars ✓";btn.style.background="#e8f0fe";} }
  if (!_synShowHL  && MAP) { _synShowHL  = true; _synHLLayer.addTo(MAP);  var btn2=document.getElementById("btn-hl");  if(btn2){btn2.textContent="H/L ✓";btn2.style.background="#e8f0fe";} }
  var styleEl = document.getElementById("syn-wx-style");
  if (styleEl) styleEl.textContent = "";
  if (_synSvgMode === "wmo") { _synSvgMode = "colour"; var btn3=document.getElementById("btn-svg"); if(btn3){btn3.textContent="Stn ✓";btn3.style.background="#e8f0fe";btn3.style.color="#1a3a6a";btn3.style.borderColor="#aaa";} var _sel=document.getElementById("ts-select"); if(_sel&&_sel.value) synUpdateTS(_sel.value); }
  var sel = document.getElementById("ts-select");
  if (sel) {
    var hh = "{EXPORT_TIME}".replace("Z","");
    var opts = Array.from(sel.options).filter(function(o){
      var v = o.value.replace(/Z$/i,"");
      return o.value.indexOf("{EXPORT_TIME}") !== -1
        || v.slice(2,4) === hh
        || v.slice(-2) === hh
        || v.indexOf(hh) !== -1;
    });
    if (opts.length) { sel.value = opts[opts.length-1].value; synUpdateTS(sel.value); }
  }
  setTimeout(synSavePNG, 800);
}
function synExportCurrent() {
  var keys = Object.keys(window).filter(function(k){return k.startsWith("map_");});
  var MAP = keys.length ? window[keys[0]] : null;
  if (!_synShowSlp && MAP) { _synShowSlp = true; _synSlpLayer.addTo(MAP); var btn=document.getElementById("btn-slp"); if(btn){btn.textContent="Isobars ✓";btn.style.background="#e8f0fe";} }
  if (!_synShowHL  && MAP) { _synShowHL  = true; _synHLLayer.addTo(MAP);  var btn2=document.getElementById("btn-hl");  if(btn2){btn2.textContent="H/L ✓";btn2.style.background="#e8f0fe";} }
  var styleEl = document.getElementById("syn-wx-style");
  if (styleEl) styleEl.textContent = "";
  if (_synSvgMode === "wmo") { _synSvgMode = "colour"; var btn3=document.getElementById("btn-svg"); if(btn3){btn3.textContent="Stn ✓";btn3.style.background="#e8f0fe";btn3.style.color="#1a3a6a";btn3.style.borderColor="#aaa";} var _sel=document.getElementById("ts-select"); if(_sel&&_sel.value) synUpdateTS(_sel.value); }
  setTimeout(synSavePNG, 400);
}
function synExportCurrentMetar() {
  var hadSlp = _synShowSlp, hadHL = _synShowHL;
  if (hadSlp) { _synShowSlp = false; _synSlpLayer.remove(); }
  if (hadHL)  { _synShowHL  = false; _synHLLayer.remove();  }
  var styleEl = document.getElementById("syn-wx-style");
  if (!styleEl) { styleEl = document.createElement("style"); styleEl.id = "syn-wx-style"; document.head.appendChild(styleEl); }
  var hadWxHidden = styleEl.textContent.indexOf("syn-wx-box") !== -1;
  styleEl.textContent = ".syn-wx-box { display: none !important; }";
  window._synMetarPNG = true;
  setTimeout(function() {
    synSavePNG();
    setTimeout(function() {
      if (hadSlp) { _synShowSlp = true; _synSlpLayer.addTo(MAP); }
      if (hadHL)  { _synShowHL  = true; _synHLLayer.addTo(MAP);  }
      if (!hadWxHidden) styleEl.textContent = "";
      window._synMetarPNG = false;
    }, 3000);
  }, 200);
}
function synToggleSvgForMetar(hide, keys) {
  var MAP = window[keys[0]];
  if (hide) {
    if (_synStnLayer) MAP.removeLayer(_synStnLayer);
  } else {
    if (_synShowSvg && _synStnLayer) _synStnLayer.addTo(MAP);
  }
}
function synExportMetar() {
  var hadSlp = _synShowSlp, hadHL = _synShowHL;
  if (hadSlp) { _synShowSlp = false; _synSlpLayer.remove(); }
  if (hadHL)  { _synShowHL  = false; _synHLLayer.remove();  }
  var styleEl = document.getElementById("syn-wx-style");
  if (!styleEl) { styleEl = document.createElement("style"); styleEl.id = "syn-wx-style"; document.head.appendChild(styleEl); }
  var hadWxHidden = styleEl.textContent.indexOf("syn-wx-box") !== -1;
  styleEl.textContent = ".syn-wx-box { display: none !important; }";
  window._synMetarPNG = true;
  setTimeout(function() {
    synSavePNG();
    setTimeout(function() {
      if (hadSlp) { _synShowSlp = true; _synSlpLayer.addTo(MAP); }
      if (hadHL)  { _synShowHL  = true; _synHLLayer.addTo(MAP);  }
      if (!hadWxHidden) styleEl.textContent = "";
      window._synMetarPNG = false;
    }, 3000);
  }, 200);
}
var _ghaPollTimer=null; var _ghaRunId=null; var _ghaTok=null;
var _ghaSteps=[
  {name:"Checkout repository",       label:"Checkout"},
  {name:"Set up Python 3.11",        label:"Setup Python"},
  {name:"Install Python packages",   label:"Install packages"},
  {name:"Determine export time",     label:"Detect export time"},
  {name:"Cache station list",        label:"Cache station CSV"},
  {name:"Generate synoptic_map.html",label:"Generate chart"},
  {name:"Publish to GitHub Pages",   label:"Publish to Pages"},
  {name:"Upload chart as artifact",  label:"Upload artifact"},
  {name:"Commit and push",           label:"Commit & push"}
];
function synShowRunPanel() {
  var p=document.getElementById("gha-panel");
  p.style.display=p.style.display==="flex"?"none":"flex";
  if(p.style.display==="flex") setTimeout(function(){document.getElementById("gha-pin").focus();},50);
  try{var _lrt=localStorage.getItem("syn_last_run");var _lrel=document.getElementById("gha-last-run");if(_lrel&&_lrt)_lrel.textContent="Last: "+_lrt;}catch(e){}
}
function synInitLastRun() {
  var _lrel=document.getElementById("gha-last-run");
  if(!_lrel) return;
  // Try localStorage first (instant)
  try{var _lrt=localStorage.getItem("syn_last_run");if(_lrt){_lrel.textContent="Last: "+_lrt;}}catch(e){}
  // Then fetch actual last run time from GitHub API (public, no token needed)
  fetch("https://api.github.com/repos/ngsmetadvisor/SfcMap/actions/workflows/synoptic_chart.yml/runs?per_page=1&status=success",{
    headers:{Accept:"application/vnd.github+json"}
  }).then(function(r){return r.json();})
  .then(function(d){
    var runs=d.workflow_runs||[];
    if(!runs.length) return;
    var _dt=new Date(runs[0].updated_at);
    var _lrt=_dt.getUTCFullYear()+"-"
      +String(_dt.getUTCMonth()+1).padStart(2,"0")+"-"
      +String(_dt.getUTCDate()).padStart(2,"0")+" "
      +String(_dt.getUTCHours()).padStart(2,"0")+":"
      +String(_dt.getUTCMinutes()).padStart(2,"0")+"Z";
    _lrel.textContent="Last: "+_lrt;
    try{localStorage.setItem("syn_last_run",_lrt);}catch(e){}
  }).catch(function(){});
}
setTimeout(synInitLastRun, 800);
function synBuildStepRows() {
  var c=document.getElementById("gha-steps"); c.innerHTML="";
  _ghaSteps.forEach(function(s,i) {
    var row=document.createElement("div");
    row.style.cssText="display:flex;align-items:center;gap:5px;";
    row.innerHTML='<span id="gha-si-'+i+'" style="font-size:11px;color:#ccc;">&#9711;</span>'
      +'<span style="color:#555;flex:1;">'+s.label+'</span>'
      +'<span id="gha-st-'+i+'" style="color:#aaa;font-size:10px;min-width:55px;text-align:right;"></span>';
    c.appendChild(row);
  });
}
function synUpdateStepIcon(i,conclusion,status) {
  var ic=document.getElementById("gha-si-"+i);
  var tl=document.getElementById("gha-st-"+i);
  if(!ic) return;
  if(conclusion==="success"){ic.innerHTML="&#10003;";ic.style.color="#1a7a2a";}
  else if(conclusion==="failure"||conclusion==="cancelled"){ic.innerHTML="&#10007;";ic.style.color="#aa2222";}
  else if(conclusion==="skipped"){ic.innerHTML="&#8212;";ic.style.color="#aaa";}
  else if(status==="in_progress"){ic.innerHTML="&#9654;";ic.style.color="#7a2acc";}
  else{ic.innerHTML="&#9711;";ic.style.color="#ccc";}
  if(tl) tl.textContent=conclusion||(status==="in_progress"?"running":"");
}
function synPollRun() {
  if(!_ghaRunId||!_ghaTok) return;
  var base="https://api.github.com/repos/ngsmetadvisor/SfcMap";
  var hdr={Authorization:"Bearer "+_ghaTok,Accept:"application/vnd.github+json"};
  fetch(base+"/actions/runs/"+_ghaRunId+"/jobs",{headers:hdr})
  .then(function(r){return r.json();})
  .then(function(d){
    var job=(d.jobs||[])[0]; if(!job) return;
    var rs=document.getElementById("gha-run-status");
    if(rs) rs.textContent=job.status+(job.conclusion?" \u2192 "+job.conclusion:"");
    var done=0;
    _ghaSteps.forEach(function(gs,i){
      var m=(job.steps||[]).find(function(s){return s.name===gs.name;});
      if(m){ synUpdateStepIcon(i,m.conclusion,m.status); if(m.conclusion&&m.conclusion!=="skipped") done++; }
    });
    var bar=document.getElementById("gha-bar");
    if(bar) bar.style.width=Math.round((done/_ghaSteps.length)*100)+"%";
    if(job.status==="completed"){
      clearInterval(_ghaPollTimer); _ghaPollTimer=null;
      if(bar&&job.conclusion==="success"){bar.style.width="100%";bar.style.background="linear-gradient(90deg,#1a7a2a,#22c55e)";}
      else if(bar) bar.style.background="#aa2222";
      var st=document.getElementById("gha-status");
      if(st&&job.conclusion==="success"){st.style.color="#1a6a2a";st.textContent="\u2713 Done! Reload to see update.";var _now=new Date();var _lrt=_now.getUTCFullYear()+"-"+String(_now.getUTCMonth()+1).padStart(2,"0")+"-"+String(_now.getUTCDate()).padStart(2,"0")+" "+String(_now.getUTCHours()).padStart(2,"0")+":"+String(_now.getUTCMinutes()).padStart(2,"0")+"Z";try{localStorage.setItem("syn_last_run",_lrt);}catch(e){}var _lrel=document.getElementById("gha-last-run");if(_lrel)_lrel.textContent="Last: "+_lrt;}
      else if(st){st.style.color="#aa2222";st.textContent="Workflow "+job.conclusion+".";}
    }
  }).catch(function(){});
}
function synTriggerGHA() {
  var pin=document.getElementById("gha-pin").value.trim();
  var st=document.getElementById("gha-status");
  if(pin.length!==4){st.style.color="#aa2222";st.textContent="Enter 4-char suffix.";return;}
  _ghaTok="ghp_5te1jZS2kbyfzeYUANY6CebGtQGpza2j"+pin;
  st.style.color="#555";st.textContent="Dispatching...";
  var base="https://api.github.com/repos/ngsmetadvisor/SfcMap";
  var hdr={Authorization:"Bearer "+_ghaTok,Accept:"application/vnd.github+json","Content-Type":"application/json"};
  fetch(base+"/actions/workflows/synoptic_chart.yml/dispatches",{
    method:"POST",headers:hdr,body:JSON.stringify({ref:"main"})
  }).then(function(r){
    if(r.status!==204){return r.text().then(function(t){st.style.color="#aa2222";st.textContent="Error "+r.status+": "+t.slice(0,100);_ghaTok=null;});}
    st.textContent="Queued \u2014 finding run...";
    document.getElementById("gha-pin").value="";
    document.getElementById("gha-progress").style.display="flex";
    synBuildStepRows();
    setTimeout(function(){
      fetch(base+"/actions/runs?event=workflow_dispatch&per_page=5",{headers:hdr})
      .then(function(r){return r.json();})
      .then(function(d){
        _ghaRunId=(d.workflow_runs||[]).length?(d.workflow_runs[0].id):null;
        if(_ghaRunId){st.textContent="";_ghaPollTimer=setInterval(synPollRun,5000);synPollRun();}
        else{st.textContent="Could not find run ID.";}
      }).catch(function(){});
    },4000);
  }).catch(function(e){st.style.color="#aa2222";st.textContent="Network error: "+e.message;_ghaTok=null;});
}
</script>
<div style="position:fixed;top:10px;right:10px;z-index:10002;display:flex;flex-direction:column;gap:6px;">
  <button onclick="synExport1200Z()" style="font-family:Courier New,monospace;font-size:12px;
    padding:5px 12px;background:{BTN_BG};border:1px solid {BTN_BDR};border-radius:5px;
    color:{BTN_CLR};cursor:pointer;font-weight:bold;">&#9928; Export {EXPORT_TIME} Analysis PNG</button>
  <button onclick="synExportMetar()" style="font-family:Courier New,monospace;font-size:12px;
    padding:5px 12px;background:#f4e8c8;border:1px solid #9a6a00;border-radius:5px;
    color:#4a3000;cursor:pointer;font-weight:bold;">&#128225; Export {EXPORT_TIME} METAR PNG</button>
  <button onclick="synExportCurrent()" style="font-family:Courier New,monospace;font-size:12px;
    padding:5px 12px;background:#d4f4c8;border:1px solid #1a6a2a;border-radius:5px;
    color:#1a3a1a;cursor:pointer;font-weight:bold;">&#9200; Export Current Timestep PNG</button>
  <button onclick="synExportCurrentMetar()" style="font-family:Courier New,monospace;font-size:12px;
    padding:5px 12px;background:#e8f4e8;border:1px solid #2a7a3a;border-radius:5px;
    color:#1a4a1a;cursor:pointer;font-weight:bold;">&#128225; Export Current Timestep METAR PNG</button>
  <button onclick="synShowRunPanel()" style="font-family:Courier New,monospace;font-size:12px;
    padding:5px 12px;background:#f0e8f8;border:1px solid #6a2a9a;border-radius:5px;
    color:#3a006a;cursor:pointer;font-weight:bold;"><span id="gha-run-btn-text">&#9881; Run Script Now</span><span id="gha-last-run" style="font-size:9px;font-weight:normal;color:#9a6acc;margin-left:6px;"></span></button>
  <div id="gha-panel" style="display:none;flex-direction:row;align-items:center;gap:6px;padding:5px 10px;
    background:#faf8ff;border:1px solid #9a6acc;border-radius:5px;">
    <span style="color:#555;font-size:11px;font-family:Courier New,monospace;">PIN</span>
    <input id="gha-pin" type="password" maxlength="4" placeholder="····"
      onkeydown="if(event.key==='Enter')synTriggerGHA()"
      style="width:52px;font-family:Courier New,monospace;font-size:12px;padding:3px 5px;
      border:1px solid #9a6acc;border-radius:3px;text-align:center;"/>
    <button onclick="synTriggerGHA()" style="padding:3px 10px;background:#7a2acc;border:none;
      border-radius:3px;color:white;cursor:pointer;font-family:Courier New,monospace;font-size:11px;font-weight:bold;">&#9889; Run</button>
    <span id="gha-status" style="color:#555;font-size:10px;font-family:Courier New,monospace;"></span>
  </div>
  <div id="gha-progress" style="display:none;flex-direction:column;gap:3px;padding:6px 10px;
    background:#faf8ff;border:1px solid #9a6acc;border-radius:5px;">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <b style="color:#3a006a;font-family:Courier New,monospace;font-size:11px;">&#128640; Workflow Progress</b>
      <span id="gha-run-status" style="color:#888;font-size:10px;font-family:Courier New,monospace;"></span>
    </div>
    <div style="background:#e8e0f0;border-radius:3px;height:6px;overflow:hidden;">
      <div id="gha-bar" style="height:100%;width:0%;background:linear-gradient(90deg,#7a2acc,#a855f7);border-radius:3px;transition:width 0.6s ease;"></div>
    </div>
    <div id="gha-steps" style="display:flex;flex-direction:column;gap:2px;font-family:Courier New,monospace;font-size:10px;"></div>
  </div>
</div>'''.replace('{BTN_BG}', _btn_bg).replace('{BTN_BDR}', _btn_bdr).replace('{BTN_CLR}', _btn_clr)


    export_js = export_js.replace('"{EXPORT_TIME}"', f'"{EXPORT_TIME}"')
    export_js = export_js.replace('{EXPORT_TIME}', EXPORT_TIME)
    html = html.replace('</body>', export_js + '</body>')
    print('Export 1200Z injected')
else:
    print('Export 1200Z already present')

# ── Auto-trigger export on load ───────────────────────────────
html = re.sub(r'<script>\s*if \(document\.readyState.*?synExport1200Z.*?</script>', '', html, flags=re.DOTALL)
if 'synAutoExport' not in html:html = re.sub(r'<script>\s*if \(document\.readyState[^<]*synExport1200Z[^<]*</script>', '', html, flags=re.DOTALL)

# ── Disable all hover tooltips ────────────────────────────────────────

# 1. Station markers tooltip
html = html.replace(
    '}).bindPopup(d.popup,{maxWidth:280,closeButton:true}).bindTooltip(d.tip).addTo(_synStnLayer);',
    '}).bindPopup(d.popup,{maxWidth:280,closeButton:true}).addTo(_synStnLayer);'
)

# 2. Isobar contour lines tooltip
html = html.replace(
    '}).bindTooltip(Math.round(ct.level)+" ").addTo(_synSlpLayer);',
    '}).addTo(_synSlpLayer);'
)

# 3. H/L markers tooltip
html = html.replace(
    '}).bindTooltip(c.type).addTo(_synHLLayer);',
    '}).addTo(_synHLLayer);'
)

print('Tooltips disabled')

with open('output/synoptic_map.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Saved')




# clear_output (no-op in headless mode)
with open('output/synoptic_map.html', encoding='utf-8') as f:
    _html = f.read()

# Build a data-URI download link so the user can save the HTML file directly.
_html_b64 = __import__('base64').b64encode(_html.encode('utf-8')).decode()
_dl_href  = f'data:text/html;base64,{_html_b64}'
# Filename = data time + 6 h (the next synoptic time = the chart label)
# e.g. data=00Z → label 06Z, data=12Z → label 18Z, data=18Z → label 00Z next day
from datetime import datetime, timedelta, timezone as _tz
_data_hour  = int(EXPORT_TIME.replace('Z', ''))   # e.g. 1200 → 1200, 1800 → 1800
_data_dt    = datetime.now(_tz.utc).replace(hour=_data_hour // 100,
                                             minute=0, second=0, microsecond=0)
_label_dt   = _data_dt + timedelta(hours=6)
_dl_name    = _label_dt.strftime('synoptic_map_%Y%m%d_%Hz') + '.html'

display(HTML(
    f'<div style="font-family:Courier New,monospace;padding:8px 10px;'
    f'background:#f8f8f8;border:1px solid #ccc;border-radius:6px 6px 0 0;'
    f'display:flex;gap:10px;align-items:center;">'
    f'<button onclick="synExport1200Z()" style="font-size:13px;padding:7px 16px;'
    f'background:{_btn_bg};border:1px solid {_btn_bdr};border-radius:5px;'
    f'color:{_btn_clr};cursor:pointer;font-weight:bold;">&#128247; Export {EXPORT_TIME} PNG</button>'
    f'<button onclick="synExportCurrent()" style="font-size:13px;padding:7px 16px;'
    f'background:#d4f4c8;border:1px solid #1a6a2a;border-radius:5px;'
    f'color:#1a3a1a;cursor:pointer;font-weight:bold;">&#128247; Export Current PNG</button>'
    f'<a href="{_dl_href}" download="{_dl_name}" '
    f'style="font-family:Courier New,monospace;font-size:13px;padding:7px 16px;'
    f'background:#e8e8f8;border:1px solid #4a4a9a;border-radius:5px;'
    f'color:#1a1a6a;cursor:pointer;font-weight:bold;text-decoration:none;">'
    f'&#11015; Download HTML</a>'
    f'<button onclick="synShowRunPanel()" '
    f'style="font-family:Courier New,monospace;font-size:13px;padding:7px 16px;'
    f'background:#f0e8f8;border:1px solid #6a2a9a;border-radius:5px;'
    f'color:#3a006a;cursor:pointer;font-weight:bold;"><span id="gha-run-btn-text">&#9881; Run Script Now</span><span id="gha-last-run" style="font-size:9px;font-weight:normal;color:#9a6acc;margin-left:6px;"></span></button>'
    f'<div id="gha-panel" style="display:none;align-items:center;gap:8px;padding:6px 12px;'
    f'background:#faf8ff;border:1px solid #9a6acc;border-radius:6px;font-family:Courier New,monospace;font-size:12px;">'
    f'<span style="color:#6a2a9a;font-weight:bold;">&#128273;</span>'
    f'<span style="color:#555;letter-spacing:0.05em;">PIN</span>'
    f'<input id="gha-pin" type="password" maxlength="4" placeholder="····" '
    f'onkeydown="if(event.key===\'Enter\')synTriggerGHA()" '
    f'style="width:54px;font-family:Courier New,monospace;font-size:13px;padding:4px 6px;'
    f'border:1px solid #9a6acc;border-radius:4px;text-align:center;letter-spacing:0.15em;"/>'
    f'<button onclick="synTriggerGHA()" '
    f'style="padding:5px 14px;background:#7a2acc;border:none;border-radius:4px;color:white;'
    f'cursor:pointer;font-family:Courier New,monospace;font-size:12px;font-weight:bold;">&#9889; Run</button>'
    f'<span id="gha-status" style="color:#555;font-size:11px;"></span>'
    f'</div>'
    f'<div id="gha-progress" style="display:none;flex-direction:column;gap:4px;padding:8px 14px;'
    f'background:#faf8ff;border:1px solid #9a6acc;border-radius:6px;font-family:Courier New,monospace;font-size:11px;min-width:380px;">'
    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">'
    f'<b style="color:#3a006a;font-size:12px;">&#128640; Workflow Progress</b>'
    f'<span id="gha-run-status" style="color:#888;font-size:10px;"></span>'
    f'</div>'
    f'<div style="background:#e8e0f0;border-radius:4px;height:8px;overflow:hidden;margin-bottom:6px;">'
    f'<div id="gha-bar" style="height:100%;width:0%;background:linear-gradient(90deg,#7a2acc,#a855f7);'
    f'border-radius:4px;transition:width 0.6s ease;"></div>'
    f'</div>'
    f'<div id="gha-steps" style="display:flex;flex-direction:column;gap:3px;"></div>'
    f'</div>'
    f'<script>'
    f'var _ghaPollTimer=null;var _ghaRunId=null;var _ghaTok=null;'
    f'var _ghaSteps=['
    f'  {{name:"Checkout repository",      label:"Checkout"}},'
    f'  {{name:"Set up Python 3.11",       label:"Setup Python"}},'
    f'  {{name:"Install Python packages",  label:"Install packages"}},'
    f'  {{name:"Determine export time",    label:"Detect export time"}},'
    f'  {{name:"Cache station list",       label:"Cache station CSV"}},'
    f'  {{name:"Generate synoptic_map.html",label:"Generate chart"}},'
    f'  {{name:"Publish to GitHub Pages",  label:"Publish to Pages"}},'
    f'  {{name:"Upload chart as artifact", label:"Upload artifact"}},'
    f'  {{name:"Commit and push",          label:"Commit & push"}}'
    f'];'
    f'function synShowRunPanel(){{'
    f'  var p=document.getElementById("gha-panel");'
    f'  p.style.display=p.style.display==="flex"?"none":"flex";'
    f'  if(p.style.display==="flex")setTimeout(function(){{document.getElementById("gha-pin").focus();}},50);'
    f'}}'
    f'function synBuildStepRows(){{'
    f'  var c=document.getElementById("gha-steps"); c.innerHTML="";'
    f'  _ghaSteps.forEach(function(s,i){{'
    f'    var row=document.createElement("div");'
    f'    row.style.cssText="display:flex;align-items:center;gap:6px;padding:2px 0;";'
    f'    row.id="gha-step-"+i;'
    f'    row.innerHTML=\'<span id="gha-si-\'+i+\'" style="font-size:13px;">&#9711;</span>\''
    f'      +\'<span style="color:#555;flex:1;">\'+s.label+\'</span>\''
    f'      +\'<span id="gha-st-\'+i+\'" style="color:#aaa;font-size:10px;min-width:60px;text-align:right;"></span>\';'
    f'    c.appendChild(row);'
    f'  }});'
    f'}}'
    f'function synUpdateStepIcon(i,conclusion,status){{'
    f'  var ic=document.getElementById("gha-si-"+i);'
    f'  var tl=document.getElementById("gha-st-"+i);'
    f'  if(!ic)return;'
    f'  if(conclusion==="success"){{ic.innerHTML="&#10003;";ic.style.color="#1a7a2a";}}'
    f'  else if(conclusion==="failure"||conclusion==="cancelled"){{ic.innerHTML="&#10007;";ic.style.color="#aa2222";}}'
    f'  else if(conclusion==="skipped"){{ic.innerHTML="&#8212;";ic.style.color="#aaa";}}'
    f'  else if(status==="in_progress"){{ic.innerHTML="&#9654;";ic.style.color="#7a2acc";}}'
    f'  else{{ic.innerHTML="&#9711;";ic.style.color="#ccc";}}'
    f'  if(tl)tl.textContent=conclusion||(status==="in_progress"?"running":"");'
    f'}}'
    f'function synPollRun(){{'
    f'  if(!_ghaRunId||!_ghaTok)return;'
    f'  var base="https://api.github.com/repos/ngsmetadvisor/SfcMap";'
    f'  var hdr={{Authorization:"Bearer "+_ghaTok,Accept:"application/vnd.github+json"}};'
    f'  fetch(base+"/actions/runs/"+_ghaRunId+"/jobs",{{headers:hdr}})'
    f'  .then(function(r){{return r.json();}})'
    f'  .then(function(d){{'
    f'    var jobs=d.jobs||[];'
    f'    var job=jobs[0];'
    f'    if(!job)return;'
    f'    var runSt=document.getElementById("gha-run-status");'
    f'    if(runSt)runSt.textContent=job.status+(job.conclusion?" \u2192 "+job.conclusion:"");'
    f'    var steps=job.steps||[];'
    f'    var done=0,total=_ghaSteps.length;'
    f'    _ghaSteps.forEach(function(gs,i){{'
    f'      var match=steps.find(function(s){{return s.name===gs.name;}});'
    f'      if(match){{'
    f'        synUpdateStepIcon(i,match.conclusion,match.status);'
    f'        if(match.conclusion&&match.conclusion!=="skipped")done++;'
    f'      }}'
    f'    }});'
    f'    var bar=document.getElementById("gha-bar");'
    f'    if(bar)bar.style.width=Math.round((done/total)*100)+"%";'
    f'    var finished=(job.status==="completed");'
    f'    if(finished){{'
    f'      clearInterval(_ghaPollTimer);_ghaPollTimer=null;'
    f'      if(bar&&job.conclusion==="success"){{bar.style.width="100%";bar.style.background="linear-gradient(90deg,#1a7a2a,#22c55e)";}}  '
    f'      if(bar&&job.conclusion!=="success"){{bar.style.background="#aa2222";}}'
    f'      var st=document.getElementById("gha-status");'
    f'      if(st&&job.conclusion==="success"){{st.style.color="#1a6a2a";st.textContent="\u2713 Done! Reload map to see update.";}}'
    f'      else if(st){{st.style.color="#aa2222";st.textContent="Workflow "+job.conclusion+".";}}'
    f'    }}'
    f'  }}).catch(function(){{}});'
    f'}}'
    f'function synTriggerGHA(){{'
    f'  var pin=document.getElementById("gha-pin").value.trim();'
    f'  var st=document.getElementById("gha-status");'
    f'  if(pin.length!==4){{st.style.color="#aa2222";st.textContent="Enter 4-char suffix.";return;}}'
    f'  _ghaTok="ghp_5te1jZS2kbyfzeYUANY6CebGtQGpza2j"+pin;'
    f'  st.style.color="#555";st.textContent="Dispatching...";'
    f'  var base="https://api.github.com/repos/ngsmetadvisor/SfcMap";'
    f'  var hdr={{Authorization:"Bearer "+_ghaTok,Accept:"application/vnd.github+json","Content-Type":"application/json"}};'
    f'  fetch(base+"/actions/workflows/synoptic_chart.yml/dispatches",{{'
    f'    method:"POST",headers:hdr,body:JSON.stringify({{ref:"main"}})'
    f'  }}).then(function(r){{'
    f'    if(r.status!==204){{return r.text().then(function(t){{st.style.color="#aa2222";st.textContent="Error "+r.status+": "+t.slice(0,120);_ghaTok=null;}});}}'
    f'    st.textContent="Queued \u2014 finding run...";'
    f'    document.getElementById("gha-pin").value="";'
    f'    document.getElementById("gha-progress").style.display="flex";'
    f'    synBuildStepRows();'
    f'    setTimeout(function(){{'
    f'      fetch(base+"/actions/runs?event=workflow_dispatch&per_page=5",{{headers:hdr}})'
    f'      .then(function(r){{return r.json();}})'
    f'      .then(function(d){{'
    f'        var runs=(d.workflow_runs||[]);'
    f'        _ghaRunId=runs.length?runs[0].id:null;'
    f'        if(_ghaRunId){{'
    f'          st.textContent="";'
    f'          _ghaPollTimer=setInterval(synPollRun,5000);'
    f'          synPollRun();'
    f'        }}else{{st.textContent="Could not find run ID \u2014 check Actions tab.";}}'
    f'      }}).catch(function(){{}});'
    f'    }},4000);'
    f'  }}).catch(function(e){{st.style.color="#aa2222";st.textContent="Network error: "+e.message;_ghaTok=null;}});'
    f'}}'
    f'</script>'
    f'</div>'
    f'<div style="width:100%;height:1800px;border:1px solid #ccc;border-radius:0 0 6px 6px;overflow:hidden;">'
    + _html +
    f'</div>'
))

print("\n✓ synoptic_map.html written to output/")
