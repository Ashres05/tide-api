# Weekly share attribution (`change-attribution`)

This branch adds a serving path that answers **why Atlantic Music Group’s
Current-US share moved this week**, names the albums behind that move, and
repeats the same week object for the other seven panel labels.

Live endpoint: `GET /v1/marketshare/attribution`

The Vercel frontend talks to this host over Cloudflare. Shipping this repo to
`main` does not deploy the API; `sudo systemctl restart tide-api` does.
Frontend layout ships in the Vercel repo.

## What was added

| Area | Files |
| --- | --- |
| Serving | `change_attribution.py`, `GET /v1/marketshare/attribution` in `api/main.py`, `model_handler.get_change_attribution` |
| Weekly pipeline | `train_model.py` stage after `releases_by_q`; S3 csv scope in `api/s3_pull.py` |
| Snowflake | `queries/query_attribution_mrelg_weekly.sql` (incremental MRELG×week AE floor) |
| τ research (offline) | `scripts/phase0_change_attribution.py`, `scripts/phase2_tau_experiment.py`, `scripts/calibrate_owner_tau.py` |
| Phase-0 SQL | `queries/query_attribution_panel_weekly.sql`, `query_attribution_icpn_case.sql`, `query_attribution_phase0_confirm.sql` |

Facts written under `model/data/` (gitignored CSVs, synced on S3 csv scope):

- `attribution_mrelg_weekly.csv` — raw all-owner MRELG×week AE (≥ 2,500)
- `amg_week_market.csv` — one AMG + market row per week
- `amg_week_owners.csv` — eight panel rows per week
- `amg_week_mrelgs.csv` — dual-gate titles
- `amg_week_labels.csv` — owner rollup of gated titles

## Market definition

The pie is **inverted Current-US album-equivalents** from `current_data_all`
(universe Current share). Do **not** sum `TARGET_LABELS` as the market.

AMG is the parent of Atlantic Records / 10K / 300 in `marketshare_weekly`.
Do not add those three on top of AMG.

Share on this endpoint is **weekly**. The market page YTD figure (e.g. ~8.52)
will not match the weekly pack (e.g. ~8.36). Join
`GET /v1/marketshare/weekly` (or actuals) if the UI needs YTD next to this
story. Attribution itself is week-over-week only.

Week dates are Luminate Thursdays (`week_end_date` / frontend
`week_ending_date`).

## Dual gate (which titles get named)

A title is listed if it is a **mover** or an **anchor** (or both).

| Gate | Rule | Plain language |
| --- | --- | --- |
| `mover` | \|ΔAE\| / this week’s market ≥ **25 bp** (`τΔ`, shared) | The album’s volume change is material vs the whole Current-US pie |
| `anchor` | AE / this week’s market ≥ that owner’s **τL** | The album is a large slice even if it barely moved |
| `both` | both rules | Large *and* moved |

`τΔ` is not per-owner. `τL` is:

| Owner | τL (bp of inverted market) | Why |
| --- | --- | --- |
| REPUBLIC Collective, Columbia Records | 75 | Held weeks already have huge titles |
| IGA | 50 | Held max ~52 bp |
| Atlantic Music Group, RCA Records | 40 | AMG lock; RCA held ~44–50 |
| Warner Records, CMG, THE ORCHARD | 25 | Held weeks top out ~27–38 bp; 40 would miss them |

Default τL for anyone not in the map is 75. Map lives in
`change_attribution.TAU_LEVEL_BY_OWNER`. Recalibrate with
`scripts/calibrate_owner_tau.py` (local replay, no Snowflake).

Title-level `d_market_bps` is **not** “this album’s share went up x bp”:

```
d_market_bps = 10000 × (this week AE − last week AE) / this week’s Current-US AE
```

New titles count the full week of AE as the change. `ae_share_bps` is **size
this week** (AE / market × 10000), not the week-over-week move.

## Drivers (the week headline)

AMG week `driver` (also `ui.driver`):

| Token | Meaning |
| --- | --- |
| `outgrown` | AMG volume up, share down — pie grew faster |
| `held` | AMG barely moved, market faded, share up on anchors |
| `other_up` | Share fell because Other (outside the eight) grew the pie |
| `amg_up` / `amg_down` | AMG share moved on AMG title gains / losses |
| `market_up` / `market_down` | Market expansion / fade dominates |
| `mix` | Own vs market effects mixed |
| `na` | Not enough lag to classify (first week in the series) |

Non-AMG owner cards use `own_up` / `own_down` in place of `amg_up` /
`amg_down`. Headlines are filled server-side (`ui.headline` for AMG,
`owners[].headline` per label).

## `GET /v1/marketshare/attribution`

```
GET /v1/marketshare/attribution
GET /v1/marketshare/attribution?week_ending_date=2026-10-08
```

Query facets filter **`mrelgs` and `labels` only**. `week` and `owners` always
describe the full Current-US pie.

| Param | Values |
| --- | --- |
| `week_ending_date` | `YYYY-MM-DD` Thursday; omit = latest complete week |
| `owner_label` | exact panel name |
| `in_panel` | `true` = eight labels, `false` = Other |
| `is_amg` | `true` = AMG titles only |
| `gate` | `mover` \| `anchor` \| `both` (`ballast` is accepted and mapped to `anchor`) |
| `tag` | `amg` \| `panel` \| `other` |

### Top-level payload

```json
{
  "config": { "tau_delta_bps": 25, "tau_level_by_owner": { "...": 40 }, "...": "..." },
  "weeks": ["2025-... Thursday", "..."],
  "week": { "... AMG decomp ..." },
  "ui": { "driver": "held", "headline": "...", "lead_rule": "...", "sort": "...", "facet_note": "..." },
  "mrelgs": [ { "... titles, already sorted ..." } ],
  "labels": [ { "... gated owner rollup ..." } ],
  "owners": [ { "... eight cards ..." } ]
}
```

`mrelgs` is **one shared title list** for the week (typically tens of rows,
p90 still under ~50). Do not clone it eight times. On an owner card, highlight
`owners[].lead_mrelg_ids`.

### `week` (AMG)

Useful fields: `amg_share`, `d_share_bps`, `own_effect_bps`,
`market_effect_bps`, `driver`, `n_titles` / `n_movers` / `n_anchors`,
`coverage_same_sign`, `coverage_mover_same_sign`.

- `d_share_bps` — AMG’s **share of Current US** moved this many bp (header number).
- `own_effect_bps` — AMG volume change expressed as bp of this week’s pie.
- `market_effect_bps` — mix: last week’s AMG share × market volume change.

Coverage percents are **logic, not display**. `coverage_same_sign` /
`coverage_mover_same_sign` are clipped 0–1 (named titles that moved with the
market). `coverage_d_market` is a net residual and can exceed 1; do not show
it. Use blurbs (strong / mixed / thin story) instead of printing percents.

### `mrelgs[]` (titles)

Already sorted for `ui.driver`. Extra fields from `annotate_titles_for_driver`:

| Field | Use |
| --- | --- |
| `section` | `lead` = proof of the headline (open); `also` = other named titles (collapsed) |
| `lead` | boolean alias of lead section |
| `tag` | `amg` / `panel` / `other` |
| `gate` | `mover` / `anchor` / `both` |
| `delta_ae`, `ae` | album-equivalents — prefer these on the row, not bp |
| `d_market_bps` | contribution to the pie; **do not label “+x bp share”** |
| `ae_share_bps` | size this week; optional muted “about 0.4% of Current US” |
| `is_new` | first sale this Luminate week |
| `owner_label` | panel name or Other |

### `owners[]` (eight cards)

Each card: `owner_label`, `share`, `d_share_bps`, `own_effect_bps`,
`market_effect_bps`, `driver`, `headline`, `coverage_level`,
`coverage_delta`, `n_titles` / `n_movers` / `n_anchors`, `tau_level_bps`,
`lead_mrelg_ids`, `tag`.

- `coverage_level` — named AE / that owner’s AE (how much of *their* catalog is on the list).
- `coverage_delta` — named \|ΔAE\| / \|owner ΔAE\|, clipped at 1.
- These are **not** market same-sign cloned eight times. Still: blurbs, not percents.

## Frontend layout (non-technical)

Keep **basis points on the week header only** (“AMG share +8 bp”). On title
rows use albums.

1. Do not re-sort `mrelgs`. Split on `section`.
2. Open **Lead** (~8 rows). Collapse **Also named**.
3. Row: art, artist — title; chip AMG / owner / “Outside the eight”; optional
   “New this week”.
4. Primary number: movers → `delta_ae` (“Up 12,000 albums”); anchors-only →
   `ae` (“48,000 albums this week”).
5. Gate chips: Moved / Held the line / Held the line · moved.
6. Do not show τ, `d_market_bps`, `ae_share_bps`, coverage, or the word
   “share” on a title row.

Lead / Also headers follow `ui.driver`:

| driver | Lead | Also |
| --- | --- | --- |
| `outgrown`, `market_up` | Grew the rest of the market | AMG this week |
| `held` | AMG titles that held | What faded in the market |
| `amg_up` / `own_up` | AMG titles that grew | Rest named this week |
| `amg_down` / `own_down` | AMG titles that fell | Rest named this week |
| `other_up` | Outside the eight, grew the market | Rest named this week |
| `market_down` | Titles that faded | Rest named this week |
| `mix`, `na` | Biggest moves this week | Rest named this week |

## Pipeline and deploy

`refresh_data` / `refresh_weekly` pulls an incremental MRELG floor ( lag of
**7 days** before the CSV max so ΔAE has a previous week — that min date is
an overlap, not a stop date), then rewrites the four `amg_week_*.csv` files
at locked τ.

After code changes on this host:

```bash
sudo systemctl restart tide-api
sudo systemctl status tide-api
```

No `--reload`. CSVs are gitignored; S3 csv scope is the replica.

## Out of scope

- YTD attribution pack (weekly marketshare endpoint already exists)
- Per-owner `τΔ`
- Dashboard chrome / Streamlit
- Treating `TARGET_LABELS` as the pie
- Printing coverage percents in the UI
