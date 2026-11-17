"""Forecast GTA VI's launch-day Twitch viewers hour by hour (hours 2-24).

Model: per-hour log-ratio curves learned from the case's own comparables
(past big game launches, 14 days before to 3 days after their release day).

For each comparable launch c and each hour h = 2..24 of its release day,
take the log-ratio L_c(h) = ln(viewers_c(h) / viewers_c(1)) against that
launch's own hour-1 print.  A launch's release-day curve is mostly a
question of scale: how many people are watching at the UK-midnight unlock
(hour 1) versus later, when the Americas unlock and evening prime time
arrives.  So for each hour we fit a kernel-weighted ridge local-linear
regression of L_c(h) on x_c = ln(viewers_c(1)) - ln(p1), where p1 is this
case's hour-1 print, with a Gaussian kernel (bandwidth 1.5 in log space).
The fitted intercept is the forecast log-ratio at this case's own scale;
the weighted residual spread around the fit sets the quantile spread.

On top of that, a per-hour additive calibration in log space (CAL, hours
2..24 below).  The donor pool mixes launch types (PC simultaneous unlocks,
early-access launches, live-service games), whose curves sit systematically
below the curve of a big console-only rolling-midnight launch like the ones
this contest grades on.  CAL is the smoothed mean log-error of this exact
model on the four public practice cases (Marvel's Wolverine, Red Dead
Redemption 2, God of War Ragnarok, Marvel's Spider-Man 2), whose hourly
answers are public SullyGnome data quoted in the kit's NOTICE.txt: it lifts
the US-unlock hours (5-7) and the evening peak (17-24), where the donor
median under-predicts that launch type, and trims hours 2-3, where it
over-predicts.  Leave-one-case-out checks (fit CAL on three practice cases,
test on the fourth) improve every held-out case, so the calibration is a
property of the launch type, not of any one game.  GTA VI is the same type
(console-only, rolling midnight local unlock, single-player), so the same
calibration applies to the final case.

The spread is recalibrated per hour as well (SPREAD_CAL, hours 2..24).
The donor residual spread is the wrong width in a systematic way: the
donor curves keep diverging over the day, so the raw spread grows with
the hour, but a real console launch's own curve is much tighter late in
the day, once the evening peak is set, and looser in the early hours,
where the unlock waves land at different times.  SPREAD_CAL is fitted
from leave-one-case-out residuals on the four practice cases: for each
hour, the ratio of the LOOCV residual spread to the model's own spread,
shrunk halfway toward 1 in log space (four cases is too few to trust a
raw per-hour ratio) and clamped to [0.6, 1.6].  It widens hours 2-5 and
12-13, where the model is over-confident, and narrows hours 19-24,
where it is far too wide.  With it, the practice loss falls from 0.1076
to 0.1004, improving every practice case (p00 0.0909 -> 0.0867,
p01 0.1292 -> 0.1244, p02 0.0736 -> 0.0671, p03 0.1367 -> 0.1233).

Quantiles are p1 * exp(centre + CAL[h] + z_tau * sd_h), with z the standard
normal quantiles at 0.05/0.25/0.5/0.75/0.95 and sd_h the residual spread
(floored at 0.05, shrunk by SPREAD_SHRINK, recalibrated by SPREAD_CAL).
A floor of 0.05 on sd and the shrink keep the bands honest without letting
one wild donor blow them up.

Python 3.13, standard library only, no network, deterministic: the same
input file always produces byte-identical output.  Reads every *.in in
$MAKEFUN_INPUT_DIR (else /input) and writes /output/<id>.out (or
$MAKEFUN_OUTPUT_DIR).  Runs in well under a second per case.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Kernel bandwidth (log units) for weighting comparables by launch scale.
BANDWIDTH = 1.5
# Ridge penalty on the local-linear slope: keeps the fit from chasing the
# one or two biggest donors when the target is far outside their range.
RIDGE = 1.0
# The residual spread is multiplied by this: the raw donor spread is wider
# than a single launch's own uncertainty once the scale is accounted for.
# Set to 0.35 (from 0.45) after the per-hour spread recalibration: with
# SPREAD_CAL the 90% band covered 89 of 92 practice hours (96.7%), i.e.
# the bands were still too wide and paid pinball loss at every hour; at
# 0.35 the coverage is 83/92 (90.2%), the honest level the scoring rule
# rewards, and the practice loss falls from 0.1004 to 0.0975.
SPREAD_SHRINK = 0.35
# Floor on the log-spread of any hour, so a lucky-tight donor set never
# claims more precision than the data supports.
SPREAD_FLOOR = 0.05

# Per-hour log-space calibration, hours 2..24 (see the module docstring for
# how it was fitted).  Positive lifts the forecast, negative trims it.
CAL = [
    -0.1761, -0.1905, -0.0502, 0.2765, 0.4622, 0.4093, 0.2909, 0.1869,
    0.1012, 0.0056, -0.0798, -0.0643, 0.0342, 0.1250, 0.2034, 0.3059,
    0.3702, 0.3417, 0.3054, 0.2958, 0.2735, 0.2437, 0.2330,
]

# Per-hour spread recalibration, hours 2..24 (see the module docstring).
# Multiplies the model spread: >1 widens, <1 narrows.
SPREAD_CAL = [
    1.273, 1.419, 1.220, 1.426, 0.792, 0.707, 0.860, 0.846,
    0.854, 1.021, 1.411, 1.247, 1.091, 0.997, 0.971, 1.036,
    0.985, 0.737, 0.681, 0.600, 0.600, 0.600, 0.678,
]

# Fallback spread (log units per sqrt-hour) when an hour has no donor data
# at all: the house baseline's widening persistence spread.
FALLBACK_SIGMA = 0.15


def parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def round2(x):
    """Round to 2 decimals, half up, returning an int for whole numbers."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def release_day_series(comp):
    """{hour 1..24: viewers} for a comparable's release UTC day."""
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    out = {}
    for point in comp.get("hourly") or []:
        try:
            when = parse_ts(point["from"])
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None or viewers <= 0:
            continue
        hour = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            out[hour] = viewers
    return out


def hour1_print(case):
    """Hour 1's print: the known row, else the last pre-checkpoint point."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers") is not None:
            return max(1, row["viewers"])
    history = case.get("history_hourly") or []
    if history:
        last = history[-1].get("viewers")
        if last:
            return max(1, last)
    return 1


def solve_linear(matrix, vector):
    """Small dense solve by Gaussian elimination with partial pivoting."""
    n = len(vector)
    rows = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(rows[r][col]))
        rows[col], rows[piv] = rows[piv], rows[col]
        if abs(rows[col][col]) < 1e-12:
            continue
        for r in range(col + 1, n):
            factor = rows[r][col] / rows[col][col]
            for cc in range(col, n + 1):
                rows[r][cc] -= factor * rows[col][cc]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = rows[i][n] - sum(rows[i][j] * out[j] for j in range(i + 1, n))
        out[i] = total / rows[i][i] if abs(rows[i][i]) > 1e-12 else 0.0
    return out


def forecast_hour(p1, hour, donors):
    """Five quantiles for one hour, from the donor log-ratio curves."""
    log_p1 = math.log(max(p1, 1))
    rows = []
    for series in donors:
        if hour in series and 1 in series:
            x = math.log(series[1]) - log_p1
            ratio = math.log(series[hour]) - math.log(series[1])
            weight = math.exp(-(x * x) / (2 * BANDWIDTH * BANDWIDTH))
            rows.append((x, ratio, weight))
    if not rows:
        return None
    total_w = sum(w for _, _, w in rows)
    mean_x = sum(w * x for x, _, w in rows) / total_w
    mean_y = sum(w * y for _, y, w in rows) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for x, _, w in rows) + RIDGE
    sxy = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in rows)
    slope = sxy / sxx if sxx else 0.0
    centre = mean_y - slope * mean_x  # prediction at x = 0 (this case)
    variance = sum(
        w * (y - (centre + slope * x)) ** 2 for x, y, w in rows
    ) / total_w
    spread = max(math.sqrt(max(variance, 0.0)), SPREAD_FLOOR) * SPREAD_SHRINK
    return centre, spread


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    p1 = hour1_print(case)
    donors = []
    for comp in case.get("comparables") or []:
        series = release_day_series(comp)
        if 1 in series and len(series) >= 20:
            donors.append(series)
    rows = []
    for row in case["forecast"]:
        hour = row["hour"]
        fit = forecast_hour(p1, hour, donors)
        if fit is None:
            # No donor covers this hour: persistence with a widening band.
            centre, spread = 0.0, FALLBACK_SIGMA * math.sqrt(hour - 1)
        else:
            centre, spread = fit
            spread *= SPREAD_CAL[hour - 2]
        centre += CAL[hour - 2]
        median = max(p1, 1) * math.exp(centre)
        quantiles = []
        previous = 0.0
        for z in Z:
            value = median * math.exp(z * spread)
            value = min(max(value, 0.01), 1_000_000_000)
            value = round2(value)
            # Rounding must not break the non-decreasing order.
            if value < previous:
                value = previous
            quantiles.append(value)
            previous = value
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
