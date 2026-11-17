# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nguyenhuong74/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nguyenhuong74/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts Grand Theft Auto VI's launch-day Twitch viewers hour by hour (hours 2–24) from the hour-1 print, using the release-day curves of 40+ past big game launches. For each hour, a kernel-weighted local-linear regression of log(viewers_h / viewers_1) on launch scale across the comparables, with a per-hour calibration fitted on the public practice cases. Python 3.13, standard library only, no network, deterministic.

The previous version of this entry scores a mean pinball loss of **0.0975** on the four public practice cases (house baseline: 0.387), with 90.2% coverage of its 90% band. This version widens the scale kernel's bandwidth from 1.5 to 2.0 log units, chosen by leave-one-out cross-validation over the 37 comparables with a complete release day: on the launches that resemble the final case, the wider bandwidth cuts the loss from 0.201 to 0.191 (rolling big launches), 0.302 to 0.271 (mega launches, hour-1 print ≥ 150k) and 0.304 to 0.284 (all large launches), improving 10 of the 12 rolling-big and mega launches. At a large hour-1 print the narrower kernel leaves too few effective donors and the local-linear slope is noisy; the wider kernel stabilises it. The change moves the practice-case median forecasts by at most 14% (Wolverine's evening hours) and at most 6% on the other three cases. See `main.py` for the full evidence, including the alternatives tested and rejected (median centre, rolling-type donor weights, empirical quantiles, channels kernel, hour smoothing, calibration/shrink changes).

## How it works

1. **Donor curves.** Each comparable launch contributes its release-day hourly curve, normalised by its own hour-1 print: `L(h) = ln(viewers_h / viewers_1)`. Launches with simultaneous global unlocks, early access, or live-service shapes stay in the pool — the kernel weighting and the calibration handle them — but only launches with a real hour-1 print and at least 20 of the 24 release-day hours count.
2. **Local-linear fit per hour.** For hour `h`, regress `L(h)` on `ln(viewers_1) - ln(p1)` (how much bigger or smaller the donor's launch was than this case's hour-1 print), weighted by a Gaussian kernel (bandwidth 2.0 in log space) with a small ridge on the slope. The intercept is the forecast log-ratio at this case's scale; the weighted residual spread sets the quantile width (shrunk ×0.35, floored at 0.05 log units, recalibrated per hour).
3. **Type calibration.** The donor pool mixes launch types, and its median curve sits systematically below the curve of a big console-only rolling-midnight launch — the type this contest grades. A smoothed per-hour log correction (fitted on the four public practice cases, documented in `main.py`) lifts the US-unlock hours (5–7) and the evening peak (17–24) and trims hours 2–3.
4. **Quantiles.** `q = p1 · exp(centre + CAL[h] + z_τ · sd)`, rounded half-up to 2 decimals, clamped to (0, 1e9], non-decreasing.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes one `<id>.out` per case: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, …]}` — exactly 23 rows, hours 2–24. No arguments, no network, no dependencies; a run takes well under a second per case and is byte-identical every time.

---

<sub>Entered on [make.fun](https://make.fun/?ref=6nfavmbn&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
