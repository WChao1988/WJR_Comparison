# WJR_Comparison — Seafloor Topography Inversion Methods

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23340061.svg)](https://doi.org/10.5281/zenodo.23340061)

Code and data accompanying **"A Review of Seafloor Topography Inversion
Methods"**. This repository packages self-contained, reproducible experiments
that benchmark six approaches for predicting seafloor topography (bathymetry)
from satellite-derived gravity fields:

| Method | Family | Included in the runnable experiments |
|---|---|---|
| **GGM** — Gravity Geologic Method | physics, spatial-domain | ✅ |
| **EGGM** — Extended GGM (prior-topography terrain correction) | physics, spatial-domain | ✅ |
| **SAS** — Smith & Sandwell-style frequency-domain inversion | physics, spectral | ✅ |
| **VParker** — iterative Parker spectral inversion | physics, spectral | ✅ |
| **PDNN** — physics-driven neural network | deep learning | reference code only |
| **Sandwell DNN** — Sandwell-style deep neural network | deep learning | reference code only |

The four physics-based methods are fully runnable from this repository. The
two neural-network methods are provided as reference implementations under
`methods/`; in the packaged experiment pipeline their rows are reused from
previously computed result CSVs, because their training depends on an internal
framework that is not part of this open release.

## Repository structure

```text
.
├── data/                     # All input data (public sources, see below)
│   ├── gebco/                #   GEBCO reference bathymetry (NetCDF, per region)
│   ├── swot_data/            #   SWOT-derived gravity grids (faa, vgg, east, north)
│   ├── ship/                 #   Ship soundings on the study grids (.npy + .txt)
│   └── longwave/             #   Pre-computed long-wave reference fields
├── example/                  # Runnable experiment entry points (self-contained)
│   ├── demo_example.py       #   Main comparison experiment
│   └── ship_uniformity_demo.py  # Sampling-uniformity experiment (depends on the main run)
├── methods/                  # Reference implementations per method
│   ├── GGM/  EGGM/  SAS/  VParker/          # physics-based methods
│   └── PDNN/  Sandwell DNN/                 # neural networks (need PyTorch)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Data

Three 3°×3° study regions in the Pacific Ocean are included:

| Region ID | Name | Bounding box (N, S, W, E) |
|---|---|---|
| `41_38_169_172` | Hawaiian–Emperor Seamount Chain | 41°N, 38°N, 169°E, 172°E |
| `22_19_176_179` | Central Pacific Seamounts | 22°N, 19°N, 176°E, 179°E |
| `18_15_155_158` | Northwest Pacific Ocean | 18°N, 15°N, 155°E, 158°E |

Per region, `data/` contains:

- `swot_data/faa_*` — free-air gravity anomaly grid (180×180, ≈1 arcmin spacing);
- `swot_data/vgg_*` — vertical gravity gradient;
- `swot_data/east_*`, `swot_data/north_*` — east/north deflection components;
- `ship/ship_*` — ship soundings gridded onto the same 180×180 mesh
  (0 = no data);
- `gebco/gebco_*.nc` — GEBCO reference bathymetry used for outlier cleaning
  and validation context;
- `longwave/longwave_*` — pre-computed long-wavelength fields (720×720).

Each `.npy` array has an equivalent human-readable `.txt` dump with the same
name. Arrays are oriented north→south (rows) and west→east (columns).

## Installation

Requires **Python ≥ 3.10**.

```bash
git clone https://github.com/WChao1988/WJR_Comparison.git
cd WJR_Comparison
pip install -r requirements.txt
```

`numba` is optional (it accelerates the EGGM terrain-correction kernel; a
pure-NumPy fallback is used when it is missing). `xarray` / `netCDF4` /
`h5py` are only needed if your GEBCO files are NetCDF-4/HDF5 — the packaged
files are NetCDF-3 classic and are read with SciPy alone.

## Usage

### 1. Main comparison experiment

```bash
cd example
python demo_example.py
```

The script, for each of the three regions and five random repeats
(seeds 11/22/33/44/55):

1. loads gravity, ship, and GEBCO grids, and applies a 5σ ship-sounding
   cleaning against GEBCO;
2. builds spatial block splits (outer train/test 90/10, inner
   fit/validation 80/20) shared by all methods;
3. tunes each physics method's parameters by grid search on the inner
   validation split;
4. evaluates on the held-out outer test split (RMSE, MAE, Bias, R², MAPE,
   Max/Min error);
5. writes all metrics, per-repeat parameters, parameter-search grids,
   prediction grids, and split masks to `example/outputs/main_comparison_repeated5/`.

Useful environment variables:

| Variable | Meaning |
|---|---|
| `BATHY_DATA_DIR` | data directory (default: `../data` relative to `example/`) |
| `BATHY_OUTPUT_DIR` | output directory (default: `example/outputs/main_comparison_repeated5`) |
| `BATHY_REGIONS_TO_RUN` | comma-separated region IDs, e.g. `18_15_155_158` |
| `BATHY_OUTER_BLOCK` / `BATHY_INNER_BLOCK` | split block sizes in pixels (auto-scaled for non-720 grids) |
| `BATHY_GEBCO_VAR` | GEBCO NetCDF variable name (default `elevation`) |

### 2. Sampling-uniformity experiment

Run **after** the main experiment (it consumes the frozen splits):

```bash
cd example
python ship_uniformity_demo.py
```

This resamples the training ship soundings at ten clustering levels
(α = 0 … 16 at a fixed retained fraction of 40%, 144-block grid), computes
sampling-uniformity metrics (Pielou U, normalized entropy H, Gini), retrains
every physics method on each resampled training set, and evaluates on the
frozen test splits. Results go to `example/outputs/uniformity_experiment/`.
Interrupted runs resume from JSON checkpoints.

## Outputs

All artifacts are written under `example/outputs/` (git-ignored):

- `metrics_all_repeats.csv` — per repeat × region × method metrics;
- `summary_by_region_method.csv`, `summary_overall_by_method.csv` —
  aggregated mean/std, macro-averaged RMSE, mean rank;
- `parameters_all_repeats.csv`, `parameter_grids/` — chosen parameters and
  full grid-search RMSE surfaces;
- `predictions/`, `shared_splits/` — full prediction grids and split masks.

## Citation

If you use this code or data, please cite:

> WCHAO. *A Review of Seafloor Topography Inversion Methods* (2026).
> Code and data: https://github.com/WChao1988/WJR_Comparison,
> DOI: 10.5281/zenodo.23340061 (archived on Zenodo).

See [`CITATION.cff`](CITATION.cff). GitHub renders a "Cite this repository"
widget from that file, and Zenodo exposes the same metadata for the archived
release.

## Acknowledgments

- GEBCO grids: GEBCO Compilation Group, *GEBCO Grid*, https://www.gebco.net
- Gravity fields derived from SWOT and satellite altimetry data
  (NASA/CNES and partners).
- Ship soundings from public bathymetric databases compiled for this study.

## License

Code is released under the [MIT License](LICENSE). The packaged data derive
from the public sources acknowledged above and are redistributed for
reproducibility; please follow the citation policies of those sources when
reusing them.
