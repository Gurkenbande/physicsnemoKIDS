#!/usr/bin/env python3
import argparse
import os
import sys

import zarr


def _decode_name(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def _open_group(path):
    try:
        return zarr.open_consolidated(path)
    except Exception:
        return zarr.open(path, mode="r")


def _print_channels(title, names):
    print(f"\n{title} ({len(names)} channels)")
    print("-" * (len(title) + 16))
    for i, name in enumerate(names):
        print(f"{i:>3}: {name}")


def _print_precip_candidates(title, names):
    keys = (
        "total_precipitation",
        "precipitation",
        "hourly_precip",
        "hourly_rain",
        "rain",
        "tp",
        "precipitable_water",
        "radar_reflectivity",
    )
    hits = []
    for i, name in enumerate(names):
        low = name.lower()
        if any(k in low for k in keys):
            hits.append((i, name))
    print(f"\n{title} precip-like candidates")
    print("-" * (len(title) + 26))
    if not hits:
        print("  (none found by keyword)")
        return
    for i, name in hits:
        print(f"{i:>3}: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Print ERA5/CWB channel names from a CWB Zarr dataset."
    )
    parser.add_argument(
        "--data_path",
        default=os.environ.get("CWA_DATA_PATH", "/tmp/cwa_dataset_3months.zarr"),
        help="Path to cwa_dataset_3months.zarr",
    )
    args = parser.parse_args()

    path = args.data_path
    if not os.path.exists(path):
        print(f"ERROR: dataset path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    g = _open_group(path)
    required = ("era5_variable", "cwb_variable")
    missing = [k for k in required if k not in g]
    if missing:
        print(
            f"ERROR: dataset is missing keys: {missing}\nAvailable keys: {list(g.keys())[:20]}",
            file=sys.stderr,
        )
        sys.exit(2)

    era5_names = [_decode_name(v) for v in g["era5_variable"][:]]
    cwb_names = [_decode_name(v) for v in g["cwb_variable"][:]]

    print(f"Dataset: {path}")
    _print_channels("ERA5 input channels", era5_names)
    _print_channels("CWB output channels", cwb_names)
    _print_precip_candidates("ERA5", era5_names)
    _print_precip_candidates("CWB", cwb_names)


if __name__ == "__main__":
    main()
