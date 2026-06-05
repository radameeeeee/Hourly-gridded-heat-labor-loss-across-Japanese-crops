"""Final one-file V2.1 Path A pipeline.

This script rebuilds the final daily Path A release from base V2.1 inputs:

1. Read layer3_v2 + layer4_sigma_hourly and rebuild fixed layer5 grid_day_task.
2. Export male/female-only daily CSV.gz files.
3. Aggregate the daily CSVs into the six small release tables.
4. Write provenance, validation CSV, and a timestamped run log.

Important methodological choices:
- sigma_day_effective is SLBS-hour-weighted and normalized by the sum of
  weights actually present on that JST day. This fixes the May-1/doy121
  false exposure caused by UTC-month boundary missing hours.
- sex == "total" is excluded from final CSVs. The final sex dimension is
  female/male only, avoiding double counting.

Default command:
    python submission_package\\code\\pipeline\\run_path_a_final_from_base.py

Fast validation of existing outputs only:
    python submission_package\\code\\pipeline\\run_path_a_final_from_base.py --validate-only
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


YEARS_DEFAULT = list(range(2010, 2025))
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KUBOTA_BASE = Path(r"C:\Users\srahd\Desktop\Kubota claude 工作")

KEEP_COLS_DAILY = [
    "year_jst",
    "doy_jst",
    "grid_id",
    "pref_code",
    "age_group",
    "sex",
    "crop_en",
    "task_en",
    "T_grid_task",
    "I_task",
    "sigma_day_effective",
    "L_task",
]

AGG_SPECS = {
    "national_year_doy": ["year_jst", "doy_jst", "age_group", "sex"],
    "pref_year_doy": ["year_jst", "pref_code", "doy_jst", "age_group", "sex"],
    "pref_year_crop": ["year_jst", "pref_code", "crop_en", "age_group", "sex"],
    "year_grid": ["year_jst", "grid_id", "age_group", "sex"],
    "year_pref_grid": ["year_jst", "pref_code", "grid_id", "age_group", "sex"],
    "national_year_crop_doy": ["year_jst", "crop_en", "doy_jst", "age_group", "sex"],
}

AGG_USECOLS = [
    "year_jst",
    "doy_jst",
    "grid_id",
    "pref_code",
    "age_group",
    "sex",
    "crop_en",
    "T_grid_task",
    "L_task",
]

AGG_DTYPES = {
    "year_jst": "int16",
    "doy_jst": "int16",
    "grid_id": "category",
    "pref_code": "int8",
    "age_group": "category",
    "sex": "category",
    "crop_en": "category",
    "T_grid_task": "float32",
    "L_task": "float32",
}


class Tee:
    """Mirror stdout/stderr to a log file while preserving console output."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    if not path.exists():
        return "NA"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_years(text: str | None) -> list[int]:
    if not text:
        return YEARS_DEFAULT
    years: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(part))
    years = sorted(set(years))
    for year in years:
        if year < 2010 or year > 2024:
            raise ValueError(f"Year out of supported range 2010-2024: {year}")
    return years


def safe_reset_dir(path: Path, required_parent: Path) -> None:
    path = path.resolve()
    required_parent = required_parent.resolve()
    if required_parent not in path.parents:
        raise RuntimeError(f"Refusing to reset unexpected path: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_slbs_f_h(kubota_base: Path) -> np.ndarray:
    slbs = kubota_base / "processed_data" / "slbs_hourly_activity_rate.csv"
    if not slbs.exists():
        return np.full(24, 1 / 24, dtype=np.float64)
    d = pd.read_csv(slbs)
    if "hour" not in d.columns or "activity_rate" not in d.columns:
        return np.full(24, 1 / 24, dtype=np.float64)
    f = (
        d.groupby("hour", as_index=True)["activity_rate"]
        .mean()
        .reindex(range(24))
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    total = float(f.sum())
    return f / total if total > 0 else np.full(24, 1 / 24, dtype=np.float64)


def hour_weighted_sigma_day(l4_year: pd.DataFrame, f_h: np.ndarray) -> pd.DataFrame:
    """Build daily effective sigma with present-hour normalization."""
    df = l4_year.copy()
    df["f_weight"] = df["hour_jst"].map({hour: float(f_h[hour]) for hour in range(24)})
    df["sigma_weighted"] = df["sigma"].astype(np.float64) * df["f_weight"]

    keys = ["year_jst", "doy_jst", "grid_id", "grid_lat", "grid_lon", "age_group"]
    daily = (
        df.groupby(keys, observed=True)
        .agg(_sig_w=("sigma_weighted", "sum"), _f_w=("f_weight", "sum"))
        .reset_index()
    )
    denom = daily["_f_w"].where(daily["_f_w"] > 0, np.nan)
    daily["sigma_day_effective"] = daily["_sig_w"] / denom
    return daily[keys + ["sigma_day_effective"]]


def build_layer5_one_year(kubota_base: Path, year: int) -> dict:
    l3_path = kubota_base / "processed_data" / "layer3_v2" / f"year={year}" / "data.parquet"
    l4_path = kubota_base / "processed_data" / "layer4_sigma_hourly" / f"year={year}" / "data.parquet"
    out_dir = kubota_base / "processed_data" / "layer5_v2" / "grid_day_task" / f"year={year}"
    out_path = out_dir / "data.parquet"
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")

    if not l3_path.exists():
        return {"year": year, "status": "MISSING_L3", "src": str(l3_path)}
    if not l4_path.exists():
        return {"year": year, "status": "MISSING_L4", "src": str(l4_path)}

    t0 = time.time()
    print(f"\n[layer5] {year}: read layer3")
    l3 = pd.read_parquet(l3_path)
    print(f"[layer5] {year}: read layer4")
    l4 = pd.read_parquet(l4_path)

    f_h = load_slbs_f_h(kubota_base)
    sigma_day = hour_weighted_sigma_day(l4, f_h)
    del l4
    gc.collect()

    print(f"[layer5] {year}: merge and compute L_task")
    merged = l3.merge(
        sigma_day,
        on=["year_jst", "doy_jst", "grid_id", "age_group"],
        how="left",
        suffixes=("", "_sigma"),
    )
    del l3, sigma_day
    gc.collect()

    for col in ["grid_lat_sigma", "grid_lon_sigma"]:
        if col in merged.columns:
            merged = merged.drop(columns=[col])

    merged["sigma_day_effective"] = merged["sigma_day_effective"].fillna(1.0)
    merged["I_task"] = (merged["T_grid_task"] > 0).astype(np.int8)
    merged["L_task"] = (
        merged["T_grid_task"]
        * merged["I_task"]
        * (1.0 - merged["sigma_day_effective"])
    )

    out_cols = [
        "year_jst",
        "doy_jst",
        "grid_id",
        "grid_lat",
        "grid_lon",
        "pref_code",
        "age_group",
        "sex",
        "crop_en",
        "task_en",
        "T_grid_task",
        "I_task",
        "sigma_day_effective",
        "L_task",
    ]
    df = merged[out_cols]
    del merged
    gc.collect()

    sigma_finite = int((~df["sigma_day_effective"].isna()).sum())
    sigma_nan = int(df["sigma_day_effective"].isna().sum())
    l_total_mh = float(df.loc[df["sex"].ne("total"), "L_task"].sum() / 1e6)
    t_total_mh = float(df.loc[df["sex"].ne("total"), "T_grid_task"].sum() / 1e6)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd", index=False)
    rows = int(len(df))
    del df
    gc.collect()

    manifest = {
        "generated_at": utc_now_iso(),
        "tool": "run_path_a_final_from_base.py",
        "model_version": "v2.1_path_a_sigma_day_fixed",
        "year_jst": year,
        "row_count": rows,
        "sigma_finite_rows": sigma_finite,
        "sigma_nan_rows": sigma_nan,
        "sex_filter_for_final_outputs": "sex != 'total'",
        "method_note": "sigma_day_effective = sum(f_h * sigma_h) / sum(f_h for hours present on the JST day)",
        "input_hashes": {
            "layer3_v2": file_sha256(l3_path),
            "layer4_sigma_hourly": file_sha256(l4_path),
        },
        "output": str(out_path),
        "output_sha256": file_sha256(out_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t0
    print(
        f"[layer5] {year}: rows={rows:,} L={l_total_mh:.3f}M "
        f"share={100*l_total_mh/t_total_mh if t_total_mh else 0:.3f}% "
        f"size={out_path.stat().st_size/1e6:.1f}MB elapsed={elapsed:.1f}s"
    )
    return {
        "year": year,
        "status": "OK",
        "rows": rows,
        "T_Mh": t_total_mh,
        "L_Mh": l_total_mh,
        "seconds": round(elapsed, 1),
        "out": str(out_path),
    }


def dump_daily_one_year(kubota_base: Path, package_root: Path, year: int) -> dict:
    src = kubota_base / "processed_data" / "layer5_v2" / "grid_day_task" / f"year={year}" / "data.parquet"
    out_dir = package_root / "data" / "path_a_daily"
    out = out_dir / f"path_a_daily_{year}.csv.gz"
    if not src.exists():
        return {"year": year, "status": "MISSING_LAYER5", "src": str(src)}

    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[daily] {year}: read layer5 parquet")
    df = pq.read_table(src, columns=KEEP_COLS_DAILY).to_pandas()
    df = df[(df["T_grid_task"] > 0) & (df["sex"] != "total")].copy()
    sex_values = sorted(df["sex"].dropna().astype(str).unique().tolist())
    if sex_values != ["female", "male"]:
        raise RuntimeError(f"Unexpected sex values in {year}: {sex_values}")

    df.to_csv(out, index=False, compression="gzip")
    rows = int(len(df))
    t_mh = float(df["T_grid_task"].sum() / 1e6)
    l_mh = float(df["L_task"].sum() / 1e6)
    share = 100.0 * l_mh / t_mh if t_mh > 0 else 0.0
    del df
    gc.collect()

    elapsed = time.time() - t0
    print(f"[daily] {year}: rows={rows:,} T={t_mh:.1f}M L={l_mh:.3f}M share={share:.3f}% elapsed={elapsed:.1f}s")
    return {
        "year": year,
        "status": "OK",
        "src": str(src),
        "src_sha256": file_sha256(src),
        "out": str(out),
        "out_sha256": file_sha256(out),
        "rows": rows,
        "national_T_Mh": t_mh,
        "national_L_Mh": l_mh,
        "national_share_pct": share,
        "seconds": round(elapsed, 1),
    }


def write_daily_manifest(package_root: Path, kubota_base: Path, years: list[int], results: list[dict]) -> None:
    out_dir = package_root / "data" / "path_a_daily"
    ok = [r for r in results if r["status"] == "OK"]
    manifest = {
        "generated_at": utc_now_iso(),
        "tool": "run_path_a_final_from_base.py",
        "source_dir": str(kubota_base / "processed_data" / "layer5_v2" / "grid_day_task"),
        "output_dir": str(out_dir),
        "columns": KEEP_COLS_DAILY,
        "row_filter": "T_grid_task > 0 and sex != 'total'",
        "sex_values": ["female", "male"],
        "years_requested": years,
        "years": results,
        "totals": {
            "rows_total": int(sum(r["rows"] for r in ok)),
            "national_L_total_Mh": float(sum(r["national_L_Mh"] for r in ok)),
            "national_T_total_Mh": float(sum(r["national_T_Mh"] for r in ok)),
        },
    }
    (out_dir / "_provenance.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def aggregate_one_year(package_root: Path, tmp_dir: Path, year: int, chunk_size: int) -> dict:
    src = package_root / "data" / "path_a_daily" / f"path_a_daily_{year}.csv.gz"
    if not src.exists():
        return {"year": year, "status": "MISSING_DAILY"}

    t0 = time.time()
    year_aggs = {name: [] for name in AGG_SPECS}
    chunks_read = 0
    rows_read = 0
    for chunk in pd.read_csv(src, usecols=AGG_USECOLS, dtype=AGG_DTYPES, chunksize=chunk_size):
        chunks_read += 1
        rows_read += int(len(chunk))
        if "total" in set(chunk["sex"].astype(str)):
            raise RuntimeError(f"sex='total' found in daily CSV for {year}")
        for name, keys in AGG_SPECS.items():
            agg = (
                chunk.groupby(keys, observed=True, as_index=False)
                .agg(T=("T_grid_task", "sum"), L=("L_task", "sum"))
            )
            year_aggs[name].append(agg)
        if chunks_read % 5 == 0:
            print(f"[agg] {year}: chunk={chunks_read} rows={rows_read/1e6:.0f}M elapsed={time.time()-t0:.0f}s")
        del chunk
        gc.collect()

    for name, parts in year_aggs.items():
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df = df.groupby(AGG_SPECS[name], observed=True, as_index=False).agg(T=("T", "sum"), L=("L", "sum"))
        df.to_csv(tmp_dir / f"_tmp_{name}_{year}.csv", index=False)
        del df, parts
        gc.collect()
    del year_aggs
    gc.collect()

    elapsed = time.time() - t0
    print(f"[agg] {year}: rows={rows_read:,} elapsed={elapsed:.1f}s")
    return {"year": year, "status": "OK", "rows": rows_read, "seconds": round(elapsed, 1)}


def concat_aggregates(package_root: Path, tmp_dir: Path, years: list[int]) -> dict:
    out_dir = package_root / "data" / "aggregates_path_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict] = {}
    print("\n[agg] combine yearly aggregate shards")
    for name, keys in AGG_SPECS.items():
        parts = []
        for year in years:
            f = tmp_dir / f"_tmp_{name}_{year}.csv"
            if f.exists():
                parts.append(pd.read_csv(f))
        if not parts:
            outputs[name] = {"status": "NO_DATA"}
            print(f"[agg] {name}: NO_DATA")
            continue
        df = pd.concat(parts, ignore_index=True)
        df = df.groupby(keys, observed=True, as_index=False).agg(T=("T", "sum"), L=("L", "sum"))
        if "sex" in df.columns and "total" in set(df["sex"].astype(str)):
            raise RuntimeError(f"sex='total' found in aggregate {name}")
        out_path = out_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        outputs[name] = {
            "status": "OK",
            "rows": int(len(df)),
            "size_mb": round(out_path.stat().st_size / 1e6, 3),
            "path": str(out_path),
            "sha256": file_sha256(out_path),
        }
        print(f"[agg] {name}: rows={len(df):,} size={out_path.stat().st_size/1e6:.2f}MB")
        del df, parts
        gc.collect()
    return outputs


def validate_outputs(kubota_base: Path, package_root: Path, years: list[int]) -> pd.DataFrame:
    """Validate layer5 parquet, daily manifest, and final aggregate totals."""
    daily_manifest_path = package_root / "data" / "path_a_daily" / "_provenance.json"
    if daily_manifest_path.exists():
        daily_manifest = json.loads(daily_manifest_path.read_text(encoding="utf-8"))
        daily_by_year = {
            int(r["year"]): r for r in daily_manifest.get("years", []) if r.get("status") == "OK"
        }
    else:
        daily_by_year = {}

    agg_path = package_root / "data" / "aggregates_path_a" / "national_year_doy.csv"
    if agg_path.exists():
        agg = pd.read_csv(agg_path)
        if "total" in set(agg["sex"].astype(str)):
            raise RuntimeError("sex='total' found in national_year_doy.csv")
        agg_y = agg.groupby("year_jst", as_index=False).agg(agg_T=("T", "sum"), agg_L=("L", "sum"))
        doy121 = (
            agg[agg["doy_jst"].eq(121)]
            .groupby("year_jst", as_index=False)
            .agg(doy121_T=("T", "sum"), doy121_L=("L", "sum"))
        )
    else:
        agg_y = pd.DataFrame(columns=["year_jst", "agg_T", "agg_L"])
        doy121 = pd.DataFrame(columns=["year_jst", "doy121_T", "doy121_L"])

    rows = []
    for year in years:
        parquet_path = kubota_base / "processed_data" / "layer5_v2" / "grid_day_task" / f"year={year}" / "data.parquet"
        layer5_t = np.nan
        layer5_l = np.nan
        layer5_rows = np.nan
        if parquet_path.exists():
            table = pq.read_table(parquet_path, columns=["sex", "T_grid_task", "L_task"])
            pdf = table.to_pandas()
            pdf = pdf[pdf["sex"].ne("total")]
            layer5_rows = int(len(pdf))
            layer5_t = float(pdf["T_grid_task"].sum())
            layer5_l = float(pdf["L_task"].fillna(0).sum())
            del pdf, table
            gc.collect()

        daily = daily_by_year.get(year, {})
        daily_t = float(daily.get("national_T_Mh", np.nan)) * 1e6
        daily_l = float(daily.get("national_L_Mh", np.nan)) * 1e6

        ay = agg_y[agg_y["year_jst"].eq(year)]
        agg_t = float(ay["agg_T"].iloc[0]) if not ay.empty else np.nan
        agg_l = float(ay["agg_L"].iloc[0]) if not ay.empty else np.nan

        d121 = doy121[doy121["year_jst"].eq(year)]
        doy121_l = float(d121["doy121_L"].iloc[0]) if not d121.empty else np.nan

        rows.append(
            {
                "year_jst": year,
                "layer5_rows_sex_not_total": layer5_rows,
                "layer5_L_Mh": layer5_l / 1e6 if not np.isnan(layer5_l) else np.nan,
                "daily_L_Mh": daily_l / 1e6 if not np.isnan(daily_l) else np.nan,
                "aggregate_L_Mh": agg_l / 1e6 if not np.isnan(agg_l) else np.nan,
                "daily_vs_layer5_rel_pct": 100.0 * (daily_l - layer5_l) / layer5_l if layer5_l else np.nan,
                "aggregate_vs_daily_rel_pct": 100.0 * (agg_l - daily_l) / daily_l if daily_l else np.nan,
                "doy121_L_Mh": doy121_l / 1e6 if not np.isnan(doy121_l) else np.nan,
                "doy121_ok": bool(abs(doy121_l) < 1e-6) if not np.isnan(doy121_l) else False,
            }
        )
    validation = pd.DataFrame(rows)
    out_path = package_root / "data" / "aggregates_path_a" / "_final_pipeline_validation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(out_path, index=False)
    print(f"\n[validate] wrote {out_path}")
    print(
        validation[
            [
                "year_jst",
                "layer5_L_Mh",
                "daily_L_Mh",
                "aggregate_L_Mh",
                "aggregate_vs_daily_rel_pct",
                "doy121_L_Mh",
            ]
        ].to_string(
            index=False,
            formatters={
                "layer5_L_Mh": "{:.3f}".format,
                "daily_L_Mh": "{:.3f}".format,
                "aggregate_L_Mh": "{:.3f}".format,
                "aggregate_vs_daily_rel_pct": "{:.6f}".format,
                "doy121_L_Mh": "{:.6f}".format,
            },
        )
    )
    max_abs_agg_diff = validation["aggregate_vs_daily_rel_pct"].abs().max()
    max_doy121 = validation["doy121_L_Mh"].abs().max()
    if max_abs_agg_diff > 0.01:
        raise RuntimeError(f"Aggregate-vs-daily validation failed: max rel pct={max_abs_agg_diff}")
    if max_doy121 > 1e-9:
        raise RuntimeError(f"doy121 validation failed: max L_Mh={max_doy121}")
    return validation


def write_final_provenance(
    package_root: Path,
    kubota_base: Path,
    years: list[int],
    layer5_results: list[dict],
    daily_results: list[dict],
    aggregate_results: list[dict],
    aggregate_outputs: dict,
    validation: pd.DataFrame,
    log_path: Path,
    started_at: str,
    elapsed_seconds: float,
) -> None:
    out = package_root / "data" / "aggregates_path_a" / "_final_pipeline_provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": utc_now_iso(),
        "started_at": started_at,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "tool": "submission_package/code/pipeline/run_path_a_final_from_base.py",
        "kubota_base": str(kubota_base),
        "package_root": str(package_root),
        "years": years,
        "method": {
            "sigma_day_effective": "sum(f_h * sigma_h) / sum(f_h for hours present on the JST day)",
            "sex_filter": "sex != 'total'",
            "final_sex_values": ["female", "male"],
            "daily_row_filter": "T_grid_task > 0 and sex != 'total'",
        },
        "outputs": {
            "layer5_grid_day_task": str(kubota_base / "processed_data" / "layer5_v2" / "grid_day_task"),
            "path_a_daily": str(package_root / "data" / "path_a_daily"),
            "aggregates_path_a": str(package_root / "data" / "aggregates_path_a"),
            "run_log": str(log_path),
        },
        "layer5_results": layer5_results,
        "daily_results": daily_results,
        "aggregate_results": aggregate_results,
        "aggregate_outputs": aggregate_outputs,
        "validation_summary": {
            "L_15yr_total_Mh": float(validation["aggregate_L_Mh"].sum()),
            "L_annual_mean_Mh": float(validation["aggregate_L_Mh"].mean()),
            "max_abs_aggregate_vs_daily_rel_pct": float(validation["aggregate_vs_daily_rel_pct"].abs().max()),
            "max_abs_doy121_L_Mh": float(validation["doy121_L_Mh"].abs().max()),
        },
    }
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[provenance] wrote {out}")


def run_pipeline(args: argparse.Namespace) -> int:
    years = parse_years(args.years)
    kubota_base = Path(args.kubota_base).resolve()
    package_root = Path(args.package_root).resolve()
    if not kubota_base.exists():
        raise FileNotFoundError(f"Kubota base not found: {kubota_base}")
    if not package_root.exists():
        raise FileNotFoundError(f"Package root not found: {package_root}")

    print("Final V2.1 Path A pipeline")
    print(f"kubota_base:  {kubota_base}")
    print(f"package_root: {package_root}")
    print(f"years:        {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"validate_only: {args.validate_only}")
    print()

    if args.validate_only:
        validation = validate_outputs(kubota_base, package_root, years)
        print(f"\n[done] validate-only L total = {validation['aggregate_L_Mh'].sum():.3f} M h")
        return 0

    started_at = utc_now_iso()
    t0 = time.time()
    layer5_results: list[dict] = []
    daily_results: list[dict] = []
    aggregate_results: list[dict] = []
    aggregate_outputs: dict = {}

    if not args.skip_layer5:
        print("\n=== Step 1/4: rebuild fixed layer5 grid_day_task ===")
        for year in years:
            result = build_layer5_one_year(kubota_base, year)
            layer5_results.append(result)
            if result["status"] != "OK":
                raise RuntimeError(f"Layer5 failed for {year}: {result}")

    if not args.skip_daily:
        print("\n=== Step 2/4: export male/female daily CSV.gz ===")
        for year in years:
            result = dump_daily_one_year(kubota_base, package_root, year)
            daily_results.append(result)
            if result["status"] != "OK":
                raise RuntimeError(f"Daily export failed for {year}: {result}")
        write_daily_manifest(package_root, kubota_base, years, daily_results)

    if not args.skip_aggregates:
        print("\n=== Step 3/4: aggregate daily CSVs into six tables ===")
        tmp_dir = package_root / "data" / "_tmp_agg_work"
        safe_reset_dir(tmp_dir, package_root / "data")
        for year in years:
            result = aggregate_one_year(package_root, tmp_dir, year, args.chunk_size)
            aggregate_results.append(result)
            if result["status"] != "OK":
                raise RuntimeError(f"Aggregation failed for {year}: {result}")
        aggregate_outputs = concat_aggregates(package_root, tmp_dir, years)
        if not args.keep_tmp:
            safe_reset_dir(tmp_dir, package_root / "data")
            tmp_dir.rmdir()

    print("\n=== Step 4/4: validate ===")
    validation = validate_outputs(kubota_base, package_root, years)
    elapsed = time.time() - t0
    write_final_provenance(
        package_root=package_root,
        kubota_base=kubota_base,
        years=years,
        layer5_results=layer5_results,
        daily_results=daily_results,
        aggregate_results=aggregate_results,
        aggregate_outputs=aggregate_outputs,
        validation=validation,
        log_path=args.log_path,
        started_at=started_at,
        elapsed_seconds=elapsed,
    )

    print("\nFinal summary")
    print(f"V2.1 Path A 15-yr total L = {validation['aggregate_L_Mh'].sum():.3f} M person-h")
    print(f"Annual mean L = {validation['aggregate_L_Mh'].mean():.3f} M person-h")
    print(f"Max aggregate-vs-daily rel diff = {validation['aggregate_vs_daily_rel_pct'].abs().max():.6f}%")
    print(f"Max doy121 L = {validation['doy121_L_Mh'].abs().max():.9f} M person-h")
    print(f"Elapsed = {elapsed/60:.1f} min")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the final V2.1 Path A pipeline from base data.")
    parser.add_argument("--kubota-base", default=str(Path(os.environ.get("KUBOTA_BASE", DEFAULT_KUBOTA_BASE))))
    parser.add_argument("--package-root", default=str(PACKAGE_ROOT))
    parser.add_argument("--years", default="2010-2024", help="Comma/range list, e.g. 2024 or 2010-2024")
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--skip-layer5", action="store_true", help="Reuse existing layer5 grid_day_task parquet")
    parser.add_argument("--skip-daily", action="store_true", help="Reuse existing path_a_daily CSV.gz files")
    parser.add_argument("--skip-aggregates", action="store_true", help="Reuse existing aggregate CSV files")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing outputs")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temporary aggregate shards")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    logs_dir = Path(args.package_root) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_path_a_final_from_base_{timestamp}.log"
    args.log_path = log_path

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(old_stdout, log_file)
        sys.stderr = Tee(old_stderr, log_file)
        try:
            return run_pipeline(args)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == "__main__":
    raise SystemExit(main())
