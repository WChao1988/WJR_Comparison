
import gc
import hashlib
import importlib.util
import json
import multiprocessing
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# 实验常量。
# --------------------------------------------------------------------------- #
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0)  # 十档聚集因子
FRAC = 0.4  # 保留点数占池比例,所有 alpha 档位一致(固定总数)
DEFAULT_RUN_REGIONS = [
    (41, 38, 169, 172),
    (22, 19, 176, 179),
    (18, 15, 155, 158),
]


def load_base_module():
    base_script = SCRIPT_DIR / "demo_example.py"
    if not base_script.exists():
        raise FileNotFoundError(f"Cannot find main script: {base_script}")
    # 按真实模块名导入(而不是 spec_from_file_location 的临时名):
    # 网格搜索的 worker 函数需要被子进程 pickle,子进程必须能按同名
    # import 到主实验模块,因此把脚本目录加入 sys.path。
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import demo_example as module
    return module


base = load_base_module()

# 主实验的冻结分块目录(重定向 base 输出之前先记录)
MAIN_OUTPUT_DIR = Path(base.OUTPUT_BASE_DIR)
SHARED_SPLITS_ROOT = MAIN_OUTPUT_DIR / "shared_splits"

OUTPUT_BASE_DIR = Path(
    os.environ.get(
        "BATHY_UNIF_OUTPUT_DIR",
        str(SCRIPT_DIR / "outputs" / "uniformity_experiment"),
    )
)
# 主实验模块的参数网格/预测产物重定向到本实验目录,避免污染原输出。
base.OUTPUT_BASE_DIR = OUTPUT_BASE_DIR
CHECKPOINT_DIR = OUTPUT_BASE_DIR / "checkpoints"

# 分块边长:默认按 12×12=144 块口径自动取网格边长/12(720 网格即 60 像元)
BLOCK_SIZE = int(os.environ.get("BATHY_UNIF_BLOCK", 0)) or None

# 可用方法 = 主实验开源版的物理方法;神经网络方法不在开源包内
ALL_METHODS = tuple(base.METHODS)


# --------------------------------------------------------------------------- #
# Windows 多进程资源防护(长实验必备)。
# --------------------------------------------------------------------------- #
MAX_POOL_WORKERS = max(1, int(os.environ.get("BATHY_MAX_POOL_WORKERS", "8")))
_POOL_CONTEXT = multiprocessing.get_context("spawn")


def _ResourceFriendlyPool(processes=None, *args, **kwargs):
    cpu = os.cpu_count() or 1
    if processes is None:
        processes = cpu
    processes = max(1, min(int(processes), MAX_POOL_WORKERS, cpu))
    kwargs.setdefault("maxtasksperchild", 64)
    return _POOL_CONTEXT.Pool(processes, *args, **kwargs)


import inspect as _inspect  # noqa: E402

# 主脚本模块级持有 Pool 符号,这里替换为资源友好版本(与原实验一致)
if getattr(base, "Pool", None) is not None:
    base.Pool = _ResourceFriendlyPool


# --------------------------------------------------------------------------- #
# 抽样与指标。
# --------------------------------------------------------------------------- #
def block_coordinates(shape, block_size):
    """边缘修正分块:与主实验 split_ship_by_blocks_cover_all 同口径。"""
    nrows, ncols = shape
    return [
        (r0, min(r0 + block_size, nrows), c0, min(c0 + block_size, ncols))
        for r0 in range(0, nrows, block_size)
        for c0 in range(0, ncols, block_size)
    ]


def block_id_map(shape, block_size):
    """每个像元的块编号(边缘块并入最后一块,即"修正编号")。"""
    nrows, ncols = shape
    n_r = -(-nrows // block_size)
    n_c = -(-ncols // block_size)
    rows_id = np.minimum(np.arange(nrows) // block_size, n_r - 1)
    cols_id = np.minimum(np.arange(ncols) // block_size, n_c - 1)
    return rows_id[:, None] * n_c + cols_id[None, :], n_r * n_c


def allocate_counts(weights, capacity, n_target, rng, max_rounds=10000):
    """迭代式配额分配:多项分布投出,超出容量的截回并在未饱和块上重投。"""
    weights = np.asarray(weights, dtype=np.float64)
    capacity = np.asarray(capacity)
    counts = np.zeros_like(capacity)
    remaining = int(n_target)
    for _ in range(max_rounds):
        if remaining <= 0:
            break
        room = capacity - counts
        w = np.where(room > 0, weights, 0.0)
        total = w.sum()
        if total <= 0:
            raise RuntimeError("no unsaturated block can take the remaining quota")
        draw = rng.multinomial(remaining, w / total)
        take = np.minimum(draw, room)
        counts += take
        remaining -= int(take.sum())
    else:
        raise RuntimeError("quota allocation did not converge")
    return counts


def uneven_resample(train_grid, alpha, frac, block_size, seed):
    """在固定总数下按聚集因子 alpha 重采样训练点,返回保留网格。"""
    rng = np.random.default_rng(seed)
    bid, n_blocks = block_id_map(train_grid.shape, block_size)
    rows, cols = np.nonzero(train_grid)
    bin_flat = bid[rows, cols]
    n_pool = int(bin_flat.shape[0])
    n_target = max(1, int(round(n_pool * frac)))
    n_b = np.bincount(bin_flat, minlength=n_blocks)
    weights = np.where(n_b > 0, n_b.astype(np.float64) ** float(alpha), 0.0)
    counts = allocate_counts(weights, n_b, n_target, rng)
    keep = np.zeros(n_pool, dtype=bool)
    for b in np.nonzero(counts)[0]:
        in_block = np.nonzero(bin_flat == b)[0]
        k = int(counts[b])
        if k > 0:
            chosen = rng.choice(in_block.shape[0], size=k, replace=False)
            keep[in_block[chosen]] = True
    resampled = np.zeros_like(train_grid, dtype=float)
    resampled[rows[keep], cols[keep]] = train_grid[rows[keep], cols[keep]]
    return resampled


def uniformity_metrics(grid, block_size):
    """五项采样侧指标,全部按 144 块口径(含空块)。"""
    bid, n_blocks = block_id_map(grid.shape, block_size)
    mask = grid != 0
    counts = np.bincount(bid[mask], minlength=n_blocks).astype(float)
    n_kept = int(counts.sum())
    s_nonempty = int(np.count_nonzero(counts))
    p = counts / max(n_kept, 1)
    pnz = p[p > 0]
    h_abs = float(-(pnz * np.log(pnz)).sum())
    h_norm = h_abs / np.log(n_blocks)
    pielou = h_abs / np.log(s_nonempty) if s_nonempty >= 2 else float("nan")
    x = np.sort(counts)
    idx = np.arange(1, len(x) + 1)
    gini = float((2 * idx - len(x) - 1).dot(x) / (len(x) * x.sum()))
    return {
        "n_kept": n_kept,
        "n_nonempty_blocks": s_nonempty,
        "U": pielou,
        "H": h_norm,
        "gini": gini,
    }


# --------------------------------------------------------------------------- #
# 检查点与指纹。
# --------------------------------------------------------------------------- #
def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialize {type(obj).__name__} to JSON")


def experiment_fingerprint():
    """含主脚本源码哈希与本实验参数;任何实现或参数改动都会作废旧检查点。"""
    main_script = SCRIPT_DIR / "demo_example.py"
    return {
        "alphas": list(ALPHAS),
        "frac": FRAC,
        "block_size": BLOCK_SIZE or "auto(=side/12)",
        "methods": [base.METHOD_LABELS[m] for m in ALL_METHODS],
        "main_script_sha256": hashlib.sha256(
            main_script.read_bytes()).hexdigest(),
        "note": "splits read from main experiment shared_splits",
    }


def checkpoint_path(region_code, repeat):
    return CHECKPOINT_DIR / region_code / f"repeat_{repeat:02d}.json"


def save_checkpoint(region_code, repeat, seed, fingerprint, entries):
    path = checkpoint_path(region_code, repeat)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "region": region_code,
        "repeat": repeat,
        "seed": int(seed),
        "config": fingerprint,
        "entries": entries,
    }
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def load_checkpoints(run_regions, repeats, fingerprint):
    entries = []
    for region in run_regions:
        region_code = base.region_id(region)
        for repeat in repeats:
            path = checkpoint_path(region_code, repeat)
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                loaded = payload["entries"]
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(f"Warning: ignoring unreadable checkpoint {path}: {error}")
                continue
            if not isinstance(loaded, list) or not loaded:
                continue
            if (
                payload.get("region") != region_code
                or payload.get("repeat") != repeat
                or payload.get("config") != fingerprint
            ):
                print(
                    f"Warning: ignoring outdated checkpoint {path} "
                    "(method code or experiment config changed)."
                )
                continue
            entries.extend(loaded)
    return entries


# --------------------------------------------------------------------------- #
# 环境过滤器与 IO。
# --------------------------------------------------------------------------- #
def _env_list(name):
    raw = os.environ.get(name, "").strip()
    return [item.strip() for item in raw.split(",") if item.strip()] if raw else []


def regions_to_run():
    codes = _env_list("BATHY_UNIF_REGIONS")
    if not codes:
        return list(DEFAULT_RUN_REGIONS)
    if codes[0].lower() == "all":
        return list(base.REGIONS)
    known = {base.region_id(r): r for r in base.REGIONS}
    unknown = [c for c in codes if c not in known]
    if unknown:
        raise ValueError(f"unknown region codes {unknown}; available {sorted(known)}")
    selected = set(codes)
    return [r for r in base.REGIONS if base.region_id(r) in selected]


def repeats_to_run():
    values = _env_list("BATHY_UNIF_REPEATS")
    if not values:
        return list(range(1, len(base.REPEAT_SEEDS) + 1))
    return sorted({int(v) for v in values})


def alphas_to_run():
    values = _env_list("BATHY_UNIF_ALPHAS")
    if not values:
        return list(ALPHAS)
    wanted = {float(v) for v in values}
    unknown = [a for a in wanted if not any(np.isclose(a, b) for b in ALPHAS)]
    if unknown:
        raise ValueError(f"alphas {unknown} not in configured levels {list(ALPHAS)}")
    return [a for a in ALPHAS if any(np.isclose(a, w) for w in wanted)]


def methods_to_run():
    values = _env_list("BATHY_UNIF_METHODS")
    if not values:
        return list(ALL_METHODS)
    unknown = [m for m in values if m not in ALL_METHODS]
    if unknown:
        print(
            f"[warn] methods {unknown} are not part of the open-source build "
            f"(neural networks excluded); available {list(ALL_METHODS)}"
        )
    selected = set(values)
    return [m for m in ALL_METHODS if m in selected]


def write_csv(path, rows, fields):
    import csv as _csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_split_files(region_code, repeat):
    """读入主实验冻结分块与测试真值,校验一致后置为只读(绝不写回)。"""
    folder = SHARED_SPLITS_ROOT / f"repeat_{repeat:02d}" / region_code
    arrays = {}
    for name in ("train", "test", "fit", "valid"):
        path = folder / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"missing frozen split file: {path}; run the main experiment first"
            )
        arrays[name] = np.load(path)
    truth_path = (
        SHARED_SPLITS_ROOT.parent / "predictions" / f"repeat_{repeat:02d}"
        / region_code / "test_truth.npy"
    )
    if truth_path.exists():
        truth = np.load(truth_path)
        if not np.array_equal(truth, arrays["test"]):
            raise AssertionError(
                f"test.npy and test_truth.npy differ at {region_code} "
                f"repeat {repeat}"
            )
    for value in arrays.values():
        value.setflags(write=False)
    return arrays


def resample_seed(seed, stable_index, alpha_index):
    """确定性种子;同一组合永远得到同一份重采样。"""
    return seed + stable_index * 100000 + (alpha_index + 1) * 1000


# --------------------------------------------------------------------------- #
# 主流程。
# --------------------------------------------------------------------------- #
SAMPLING_FIELDS = [
    "region", "repeat", "seed", "alpha",
    "n_kept", "n_nonempty_blocks", "U", "H", "gini",
]
DETAIL_FIELDS = [
    "region", "repeat", "seed", "alpha", "method",
    "n_kept", "n_nonempty_blocks", "U", "H", "gini",
    "n_train_samples", "n_fit_samples", "n_valid_samples", "n_samples",
    "RMSE", "MAE", "Bias", "R2", "MAPE", "Max_Error", "Min_Error",
]


def run_method_with_retry(method, data, split, seed, run_tag, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        try:
            return base.run_method(method, data, split, seed, run_tag=run_tag)
        except OSError as error:
            if attempt >= max_attempts:
                raise
            wait = 60 * attempt
            print(f"      Warning: {error}; retry in {wait}s "
                  f"({attempt}/{max_attempts})")
            gc.collect()
            time.sleep(wait)
            gc.collect()


def main():
    run_regions = regions_to_run()
    repeats = repeats_to_run()
    alphas = alphas_to_run()
    methods = methods_to_run()

    # 自动确定分块边长(12×12=144 块口径)
    global BLOCK_SIZE
    if BLOCK_SIZE is None:
        probe = load_split_files(base.region_id(run_regions[0]), repeats[0])
        BLOCK_SIZE = max(1, probe["train"].shape[0] // 12)
        del probe
        print(f"auto block size = {BLOCK_SIZE} px "
              f"({12}x{12} blocks on this grid)")

    fingerprint = experiment_fingerprint()
    stable_positions = {
        base.region_id(r): i for i, r in enumerate(base.REGIONS, start=1)
    }

    print("=" * 72)
    print("船测均匀度实验 (uniformity experiment, open-source build)")
    print(f"Regions  : {[base.region_id(r) for r in run_regions]}")
    print(f"Repeats  : {repeats}")
    print(f"Alphas   : {alphas}")
    print(f"Methods  : {[base.METHOD_LABELS[m] for m in methods]}")
    print(f"FRAC={FRAC}, block={BLOCK_SIZE}px (144 blocks), "
          f"pool={SHARED_SPLITS_ROOT}")
    print(f"Output   : {OUTPUT_BASE_DIR}")
    print("=" * 72)
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE_DIR / "uniformity_experiment_config.json").write_text(
        json.dumps(
            {
                "regions": [base.region_id(r) for r in run_regions],
                "repeats": repeats,
                "alphas": list(alphas),
                "frac": FRAC,
                "block_size": BLOCK_SIZE,
                "n_blocks": 144,
                "methods": [base.METHOD_LABELS[m] for m in methods],
                "fingerprint": fingerprint,
                "note": (
                    "Fixed retained count; alpha controls spatial clustering "
                    "over 144 blocks. Test set frozen from main experiment "
                    "shared_splits. Neural methods are excluded from the "
                    "open-source build."
                ),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    if os.environ.get("BATHY_FORCE_RERUN", "").strip().lower() in ("1", "true", "yes"):
        if CHECKPOINT_DIR.exists():
            shutil.rmtree(CHECKPOINT_DIR)
            print("BATHY_FORCE_RERUN: removed checkpoints")

    # ---- 第一步+第二步+第四步:重采样与采样侧指标(不依赖模型)。 ----
    sampling_rows = []
    sampling_path = OUTPUT_BASE_DIR / "uniformity_sampling_metrics.csv"
    for region in run_regions:
        region_code = base.region_id(region)
        stable_index = stable_positions[region_code]
        for repeat in repeats:
            seed = base.REPEAT_SEEDS[repeat - 1]
            splits = load_split_files(region_code, repeat)
            for alpha_index, alpha in enumerate(alphas):
                resampled = uneven_resample(
                    splits["train"], alpha, FRAC, BLOCK_SIZE,
                    resample_seed(seed, stable_index, alpha_index),
                )
                row = {
                    "region": region_code,
                    "repeat": repeat,
                    "seed": seed,
                    "alpha": alpha,
                    **uniformity_metrics(resampled, BLOCK_SIZE),
                }
                sampling_rows.append(row)
            del splits
        write_csv(sampling_path, sampling_rows, SAMPLING_FIELDS)
        print(f"[sampling] {region_code}: {len(sampling_rows)} rows so far")
    sampling_lookup = {
        (r["region"], r["repeat"], r["alpha"]): r for r in sampling_rows
    }
    print(f"[sampling] finished: {len(sampling_rows)} training sets characterised")

    # ---- 第三步+第五步:重训方法并在冻结测试集上评测。 ----
    entries = load_checkpoints(run_regions, repeats, fingerprint)
    done_keys = {
        (e["region"], e["repeat"], e["alpha"], e["method"]) for e in entries
    }
    runs_per_repeat = len(alphas) * len(methods)
    pair_counts = defaultdict(int)
    for region_code, repeat, _alpha, _method in done_keys:
        pair_counts[(region_code, repeat)] += 1
    if done_keys:
        print(f"[train] Resuming: {len(done_keys)} method runs already saved")
    else:
        print("[train] No usable checkpoints; computing from scratch.")

    records = []
    detail_path = OUTPUT_BASE_DIR / "uniformity_metrics_all.csv"

    def refresh_detail():
        order = {m: i for i, m in enumerate(
            base.METHOD_LABELS[k] for k in base.METHODS)}
        records.sort(
            key=lambda r: (
                r["region"], r["repeat"], r["alpha"],
                order.get(r["method"], 99),
            )
        )
        write_csv(detail_path, records, DETAIL_FIELDS)

    for region_index, region in enumerate(run_regions, start=1):
        region_code = base.region_id(region)
        if all(
            pair_counts[(region_code, repeat)] >= runs_per_repeat
            for repeat in repeats
        ):
            print(f"\n[train {region_index}/{len(run_regions)}] {region_code}: "
                  f"checkpoints complete, skipping")
            continue
        print(f"\n[train {region_index}/{len(run_regions)}] Preparing region {region}")
        stable_index = stable_positions[region_code]
        data = base.prepare_region(region)
        for repeat in repeats:
            seed = base.REPEAT_SEEDS[repeat - 1]
            if pair_counts[(region_code, repeat)] >= runs_per_repeat:
                print(f"  Repeat {repeat}: checkpoint complete, skipping")
                continue
            print(f"  Repeat {repeat}, seed={seed}")
            splits = load_split_files(region_code, repeat)
            pair_entries = [
                dict(e) for e in entries
                if e["region"] == region_code and e["repeat"] == repeat
            ]
            for alpha_index, alpha in enumerate(alphas):
                resampled = uneven_resample(
                    splits["train"], alpha, FRAC, BLOCK_SIZE,
                    resample_seed(seed, stable_index, alpha_index),
                )
                metrics_side = sampling_lookup[(region_code, repeat, alpha)]
                split = {
                    "train": resampled,
                    "test": splits["test"],
                    # 内层 fit/valid 块位置不动,只换块内内容。
                    "fit": np.where(splits["fit"] != 0, resampled, 0.0),
                    "valid": np.where(splits["valid"] != 0, resampled, 0.0),
                }
                n_fit = int(np.count_nonzero(split["fit"]))
                n_valid = int(np.count_nonzero(split["valid"]))
                if n_fit == 0 or n_valid == 0:
                    raise RuntimeError(
                        f"empty inner fit/valid at region={region_code}, "
                        f"repeat={repeat}, alpha={alpha:g}; reduce max alpha."
                    )
                print(
                    f"    alpha={alpha:g}: kept={metrics_side['n_kept']}, "
                    f"gini={metrics_side['gini']:.3f}, H={metrics_side['H']:.3f}"
                )
                for method in methods:
                    label = base.METHOD_LABELS[method]
                    if (region_code, repeat, alpha, label) in done_keys:
                        print(f"      {label}: checkpoint found, skipping")
                        continue
                    run_tag = f"repeat_{repeat:02d}_seed_{seed}_alpha_{alpha:g}"
                    started = time.time()
                    print(f"      Running {label}")
                    result, _prediction, params = run_method_with_retry(
                        method, data, split, seed, run_tag
                    )
                    result.update(
                        {
                            "region": region_code,
                            "repeat": repeat,
                            "seed": seed,
                            "alpha": alpha,
                            "method": label,
                            "n_train_samples": int(
                                np.count_nonzero(split["train"])
                            ),
                            "n_fit_samples": n_fit,
                            "n_valid_samples": n_valid,
                            "n_kept": metrics_side["n_kept"],
                            "n_nonempty_blocks":
                                metrics_side["n_nonempty_blocks"],
                            "U": metrics_side["U"],
                            "H": metrics_side["H"],
                            "gini": metrics_side["gini"],
                        }
                    )
                    entry = {
                        "region": region_code,
                        "repeat": repeat,
                        "alpha": alpha,
                        "method": label,
                        "result": result,
                        "params": params,
                    }
                    pair_entries.append(entry)
                    entries.append(entry)
                    done_keys.add((region_code, repeat, alpha, label))
                    pair_counts[(region_code, repeat)] += 1
                    records.append(result)
                    save_checkpoint(
                        region_code, repeat, seed, fingerprint, pair_entries
                    )
                    refresh_detail()
                    print(
                        f"      {label} finished in {time.time() - started:.1f}s"
                    )
                    gc.collect()
                    time.sleep(1.0)
            del splits

    # ---- 第六步:聚合。 ----
    refresh_detail()
    by_region = []
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["region"], row["method"], row["alpha"])].append(row)
    for (region_code, label, alpha), rows in sorted(
        grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])
    ):
        rmse = np.asarray([r["RMSE"] for r in rows], dtype=float)
        by_region.append(
            {
                "region": region_code,
                "method": label,
                "alpha": alpha,
                "n_repeats": len(rows),
                "RMSE_mean": float(np.nanmean(rmse)),
                "RMSE_std": (
                    float(np.nanstd(rmse, ddof=1)) if len(rmse) > 1 else np.nan
                ),
                "gini_mean": float(np.nanmean([r["gini"] for r in rows])),
                "H_mean": float(np.nanmean([r["H"] for r in rows])),
                "U_mean": float(np.nanmean([r["U"] for r in rows])),
            }
        )
    write_csv(
        OUTPUT_BASE_DIR / "uniformity_by_region_method_alpha.csv",
        by_region,
        [
            "region", "method", "alpha", "n_repeats",
            "RMSE_mean", "RMSE_std", "gini_mean", "H_mean", "U_mean",
        ],
    )
    macro = []
    macro_grouped = defaultdict(list)
    for row in records:
        macro_grouped[(row["method"], row["alpha"])].append(row)
    for (label, alpha), rows in sorted(
        macro_grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])
    ):
        rmse = np.asarray([r["RMSE"] for r in rows], dtype=float)
        region_means = np.asarray(
            [
                np.nanmean(
                    [r["RMSE"] for r in rows if r["region"] == region_code]
                )
                for region_code in {r["region"] for r in rows}
            ]
        )
        macro.append(
            {
                "method": label,
                "alpha": alpha,
                "n_runs": len(rows),
                "gini_mean": float(np.nanmean([r["gini"] for r in rows])),
                "H_mean": float(np.nanmean([r["H"] for r in rows])),
                "U_mean": float(np.nanmean([r["U"] for r in rows])),
                "n_kept_mean": float(np.nanmean([r["n_kept"] for r in rows])),
                "RMSE_macro_mean": float(np.nanmean(rmse)),
                "RMSE_pooled_std": (
                    float(np.nanstd(rmse, ddof=1)) if len(rmse) > 1 else np.nan
                ),
                "RMSE_region_macro_std": (
                    float(np.nanstd(region_means, ddof=1))
                    if len(region_means) > 1
                    else np.nan
                ),
                "MAE_macro_mean": float(np.nanmean([r["MAE"] for r in rows])),
                "Bias_macro_mean": float(np.nanmean([r["Bias"] for r in rows])),
            }
        )
    write_csv(
        OUTPUT_BASE_DIR / "uniformity_macro_by_method_alpha.csv",
        macro,
        [
            "method", "alpha", "n_runs", "gini_mean", "H_mean", "U_mean",
            "n_kept_mean", "RMSE_macro_mean", "RMSE_pooled_std",
            "RMSE_region_macro_std", "MAE_macro_mean", "Bias_macro_mean",
        ],
    )
    print("\nFinished.")
    print(f"Sampling rows : {len(sampling_rows)}")
    print(f"Detail rows   : {len(records)}")
    print(f"Results       : {OUTPUT_BASE_DIR}")
    print(f"Checkpoints   : {CHECKPOINT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Finished runs are checkpointed; "
            "rerun this script to resume."
        )
        sys.exit(130)
