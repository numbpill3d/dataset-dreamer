# dataset-dreamer

> generative data augmentation conditioned on external signals. tabular / timeseries synthesis with cosmic rhythm awareness.

## aesthetic

90s / 00s Aero mono Y2K but brutalist minimalist: near-black surfaces (`#0a0a0c`), muted violet (`#7b5cff`) accents, 1px black borders, no rounded corners, sharp geometric edges, monospace everywhere (`JetBrains Mono`, `VT323`). System 6 / 7 control aesthetic — no neon, no CRT dashboard skin, no neon gradients. Organization XIII sigil geometry (13-point wireframe anchors) embedded in the latent space visualization. WebGPU-powered point-sprite rendering with phosphor glow, chromatic aberration, and scanline effects.

## architecture

```
python core          rust/wasm web        cli / api
     |                   |                  |
[ ctgan / tvae  ] → [ web ui (fastapi) ] → [ dreamer cli ]
     |                   |                  |
[ custom flow  ] ← [ webgpu shader  ] ← [ signals fetch ]
     |                   |                  |
[ conditioning ] ← [ signal fetchers ] ← [ external api ]
                        moon / solar / git / crypto
```

## quick start

```bash
# install
pip install -e .

# train
python -m dataset_dreamer.cli.main train data.csv --model ctgan --epochs 300 --signal moon_phase --signal solar_wind

# generate
python -m dataset_dreamer.cli.main generate models/latest/ --rows 1000 --temp 1.2 --format parquet

# ui
python -m dataset_dreamer.ui.server --host 127.0.0.1 --port 8080
```

## core modules

- `core/synthesizer.py` — CTGAN, TVAE, and custom conditional normalizing flow
- `core/pipeline.py` — DatasetDreamer orchestration: load → fetch signals → train → generate → save
- `core/config.py` — pydantic config models
- `signals/fetchers.py` — external signal fetchers: moon phase, solar wind, git activity, crypto volatility
- `cli/main.py` — typer CLI with banner, tables, progress bars
- `ui/` — fastapi server + 90s aero brutalist UI
- `shaders/latent.wgsl` — WebGPU shaders for point-sprite rendering with phosphor glow, wireframe sigils, chromatic aberration

## ui aesthetic rules

- monospace only (`JetBrains Mono`, `VT323`)
- 1px borders (`#1a1a20` / `#2a2a34`), never rounded
- near-black background, violet (`#7b5cff`) accent only
- glass = `rgba(26,26,32,0.85)`, never frosted white
- no gradients except title bar (violet stripe) and progress bars
- 13-point sigil geometry in latent space visualization
- keyboard: `ctrl+o` load / `f5` train / `f9` generate / `f10` interpolate / `ctrl+` console / `esc` close modal

## data formats

training: `.parquet`, `.csv`, `.jsonl`, `.xlsx`, `.feather`
output: `.parquet`, `.csv`, `.jsonl`, `.polars`
model: directory with `meta.json` + weights

## signals

conditioned on 4 external rhythms:
- `moon_phase` — lunar cycle (0..1)
- `solar_wind` — solar wind speed (normalized)
- `git_activity` — commit frequency / diff size
- `crypto_volatility` — price variance metric

signals fetched synchronously at load; conditioning vector tiled to data rows.

## shaders

- `latent_vertex` — animated vertex shader with signal-driven noise displacement
- `point_sprite_fragment` — phosphor glow, HSV from latent coords, pulse from time + idx
- `wireframe_fragment` — 13-point sigil wireframe with scanline modulation
- `post_process` — CRT scanline, chromatic aberration, vignette, film grain

## build / deploy

```bash
# package
python -m build
# or for itch.io
python scripts/package_itch.py
```

## dependencies

- `ctgan` / `sdv` — generative modeling
- `pydantic` — config
- `polars` / `pandas` — data
- `fastapi` / `uvicorn` — web ui server
- `numpy` — arrays / conditioning
- `rich` — terminal UI (cli banner, progress, tables)
- `typer` — cli framework
- `torch` — custom flow
- `webgpu` — shader rendering (browser-native)

## notes

- no electron. native linux + browser webgpu.
- no rounded corners. 1px borders only.
- no placeholder categories. all names are real (moon, solar, git, crypto).
- no neon / CRT dashboard. violet + near-black only.
- direct execution mode: build first, ask later.
