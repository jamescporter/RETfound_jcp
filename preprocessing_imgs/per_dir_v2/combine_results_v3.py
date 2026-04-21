import argparse
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Canonical stats combiner for per_dir_v2.")
    parser.add_argument("--image-stats-glob", default=r"D:\2024_PHD-DATA_CFP_PROCESSING_v2\*_image_statistics.csv", help="Glob for v2 per-image stats CSV files.")
    parser.add_argument("--image-output", default=r"D:\2024_PHD-DATA_CFP_PROCESSING_v2\final_image_statistics.csv", help="Output path for combined v2 per-image stats.")
    parser.add_argument("--skip-image-combine", action="store_true", help="Skip default v3 per-image combine.")
    parser.add_argument("--include-global-stats", action="store_true", help="Optionally combine v2 *_global_stats.csv files.")
    parser.add_argument("--global-stats-glob", default=r"D:\2024_PHD-DATA_CFP_PROCESSING_v2\*_global_stats.csv", help="Glob for v2 global stats files.")
    parser.add_argument("--global-output", default=r"D:\2024_PHD-DATA_CFP_PROCESSING_v2\final_full_stats.csv", help="Output path for combined global stats.")
    parser.add_argument("--include-legacy-checkpoints", action="store_true", help="Optionally combine legacy checkpoint CSVs from v1/v2 flow.")
    parser.add_argument("--legacy-checkpoint-glob", default=r"D:\2024_PHD-DATA_CFP_PROCESSING\*image_statistics_checkpoint.csv", help="Glob for legacy checkpoint CSV files.")
    parser.add_argument("--legacy-output", default=r"D:\2024_PHD-DATA_CFP_PROCESSING_v2\final_image_statistics_legacy.csv", help="Output path for combined legacy checkpoint rows.")
    parser.add_argument("--split-legacy-rgb-columns", action="store_true", help="Optionally split legacy array-like RGB columns into _R/_G/_B columns.")
    parser.add_argument("--include-legacy-global", action="store_true", help="Optionally compute legacy global mean/std from checkpoint columns.")
    parser.add_argument("--legacy-global-output", default=r"D:\2024_PHD-DATA_CFP_PROCESSING_v2\final_full_stats_legacy.csv", help="Output path for legacy global summary.")
    return parser.parse_args()


def compute_final_stats(sum_pixels, sum_squares, total_pixels):
    global_mean = sum_pixels / total_pixels
    global_variance = (sum_squares / total_pixels) - (global_mean ** 2)
    global_std_dev = np.sqrt(global_variance)
    return global_mean, global_variance, global_std_dev


def combine_v2_image_stats(image_stats_glob, image_output):
    image_stats_files = sorted(glob.glob(image_stats_glob))
    if not image_stats_files:
        raise ValueError(f"No image stats files matched: {image_stats_glob}")
    frames = [pd.read_csv(file) for file in tqdm(image_stats_files, desc="Combining v2 image stats", unit="file")]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(image_output, index=False)
    print(f"Wrote combined v2 image stats: {image_output} ({len(combined)} rows)")


def combine_v2_global_stats(global_stats_glob, global_output):
    stats_files = sorted(glob.glob(global_stats_glob))
    if not stats_files:
        raise ValueError(f"No global stats files matched: {global_stats_glob}")
    global_sum_pixels = np.zeros(3, dtype=np.float64)
    global_sum_squares = np.zeros(3, dtype=np.float64)
    global_min_pixel = np.full(3, 255, dtype=np.float64)
    global_max_pixel = np.zeros(3, dtype=np.float64)
    global_total_pixels = 0.0
    for file in tqdm(stats_files, desc="Combining global stats", unit="file"):
        stats = pd.read_csv(file).iloc[0]
        global_sum_pixels += np.array([stats["global_sum_pixels_R"], stats["global_sum_pixels_G"], stats["global_sum_pixels_B"]])
        global_sum_squares += np.array([stats["global_sum_squares_R"], stats["global_sum_squares_G"], stats["global_sum_squares_B"]])
        global_min_pixel = np.minimum(global_min_pixel, np.array([stats["global_min_pixel_R"], stats["global_min_pixel_G"], stats["global_min_pixel_B"]]))
        global_max_pixel = np.maximum(global_max_pixel, np.array([stats["global_max_pixel_R"], stats["global_max_pixel_G"], stats["global_max_pixel_B"]]))
        global_total_pixels += stats["global_total_pixels"]
    global_mean, global_variance, global_std_dev = compute_final_stats(global_sum_pixels, global_sum_squares, global_total_pixels)
    global_stats = pd.DataFrame({
        "global_mean_R": [global_mean[0]], "global_mean_G": [global_mean[1]], "global_mean_B": [global_mean[2]],
        "global_variance_R": [global_variance[0]], "global_variance_G": [global_variance[1]], "global_variance_B": [global_variance[2]],
        "global_std_dev_R": [global_std_dev[0]], "global_std_dev_G": [global_std_dev[1]], "global_std_dev_B": [global_std_dev[2]],
        "global_min_pixel_R": [global_min_pixel[0]], "global_min_pixel_G": [global_min_pixel[1]], "global_min_pixel_B": [global_min_pixel[2]],
        "global_max_pixel_R": [global_max_pixel[0]], "global_max_pixel_G": [global_max_pixel[1]], "global_max_pixel_B": [global_max_pixel[2]],
        "global_total_pixels": [global_total_pixels],
    })
    global_stats.to_csv(global_output, index=False)
    print(f"Wrote combined v2 global stats: {global_output}")


def _split_legacy_array_columns(df, columns):
    for column in columns:
        if column not in df.columns:
            continue
        cleaned = df[column].astype(str).str.strip("[]")
        split = cleaned.str.split(expand=True)
        if split.shape[1] < 3:
            continue
        df[f"{column}_R"] = pd.to_numeric(split[0], errors="coerce")
        df[f"{column}_G"] = pd.to_numeric(split[1], errors="coerce")
        df[f"{column}_B"] = pd.to_numeric(split[2], errors="coerce")
        df.drop(columns=[column], inplace=True)


def combine_legacy_checkpoints(legacy_checkpoint_glob, legacy_output, split_legacy_rgb_columns, include_legacy_global, legacy_global_output):
    legacy_files = sorted(glob.glob(legacy_checkpoint_glob))
    if not legacy_files:
        raise ValueError(f"No legacy checkpoint files matched: {legacy_checkpoint_glob}")
    frames = [pd.read_csv(file, index_col=0) for file in tqdm(legacy_files, desc="Combining legacy checkpoints", unit="file")]
    legacy_df = pd.concat(frames)
    if split_legacy_rgb_columns:
        _split_legacy_array_columns(legacy_df, ["median", "min_pixel", "max_pixel", "skewness", "kurtosis"])
    legacy_df.to_csv(legacy_output)
    print(f"Wrote combined legacy checkpoint rows: {legacy_output} ({len(legacy_df)} rows)")
    if include_legacy_global:
        required = {"sum_pixels", "sum_squared_pixels", "total_pixels"}
        if required.issubset(legacy_df.columns):
            sum_pixels = pd.to_numeric(legacy_df["sum_pixels"], errors="coerce").sum()
            sum_squared_pixels = pd.to_numeric(legacy_df["sum_squared_pixels"], errors="coerce").sum()
            total_pixels = pd.to_numeric(legacy_df["total_pixels"], errors="coerce").sum()
            global_mean = sum_pixels / total_pixels
            global_variance = (sum_squared_pixels / total_pixels) - (global_mean ** 2)
            global_std_dev = np.sqrt(global_variance)
            pd.DataFrame([{"global_mean": global_mean, "global_variance": global_variance, "global_std_dev": global_std_dev, "global_total_pixels": total_pixels}]).to_csv(legacy_global_output, index=False)
            print(f"Wrote legacy global summary: {legacy_global_output}")
        else:
            print("Skipped legacy global summary: required columns not found.")


def main():
    args = parse_args()
    if not args.skip_image_combine:
        combine_v2_image_stats(args.image_stats_glob, args.image_output)
    if args.include_global_stats:
        combine_v2_global_stats(args.global_stats_glob, args.global_output)
    if args.include_legacy_checkpoints:
        combine_legacy_checkpoints(args.legacy_checkpoint_glob, args.legacy_output, args.split_legacy_rgb_columns, args.include_legacy_global, args.legacy_global_output)


if __name__ == "__main__":
    main()
