import argparse
import gc
import logging
import os

import cv2
import numpy as np
import pandas as pd
from numba import njit, prange
from PIL import Image
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from tqdm import tqdm


@njit(parallel=True)
def compute_intermediate_stats(img):
    sum_pixels = np.zeros(3, dtype=np.float64)
    sum_squares = np.zeros(3, dtype=np.float64)
    min_pixel = np.full(3, 255, dtype=np.float64)
    max_pixel = np.zeros(3, dtype=np.float64)
    total_pixels = img.shape[0] * img.shape[1]
    for i in prange(img.shape[0]):
        for j in prange(img.shape[1]):
            for k in range(3):
                pixel_value = img[i, j, k]
                sum_pixels[k] += pixel_value
                sum_squares[k] += pixel_value ** 2
                if pixel_value < min_pixel[k]:
                    min_pixel[k] = pixel_value
                if pixel_value > max_pixel[k]:
                    max_pixel[k] = pixel_value
    return sum_pixels, sum_squares, min_pixel, max_pixel, total_pixels


def parse_args():
    parser = argparse.ArgumentParser(description="Compute per-shard CFP image stats.")
    parser.add_argument("root_folder", help="Folder containing PNG files (walked recursively).")
    parser.add_argument("--output-folder", default=r"D:\2024_PHD-DATA_CFP_PROCESSING_v2", help="Directory for output CSV and log files.")
    parser.add_argument("--write-image-stats", action="store_true", help="Also write per-image stats CSV (default off).")
    parser.add_argument("--legacy-v1-metrics", action="store_true", help="When writing per-image stats, include extra v1 metrics (default off).")
    return parser.parse_args()


def setup_logging(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_file_path)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def build_basic_image_stats(img):
    median = np.median(img, axis=(0, 1))
    min_pixel = img.min(axis=(0, 1))
    max_pixel = img.max(axis=(0, 1))
    return {
        "median_R": median[0], "median_G": median[1], "median_B": median[2],
        "min_pixel_R": min_pixel[0], "min_pixel_G": min_pixel[1], "min_pixel_B": min_pixel[2],
        "max_pixel_R": max_pixel[0], "max_pixel_G": max_pixel[1], "max_pixel_B": max_pixel[2],
    }


def build_legacy_v1_metrics(img):
    img_flat = img.reshape(-1, 3)
    img_skewness = skew(img_flat, axis=0)
    img_kurtosis = kurtosis(img_flat, axis=0)
    img_entropy = shannon_entropy(img)
    img_gray = np.mean(img, axis=2).astype(np.uint8)
    img_gray_resized = cv2.resize(img_gray, (256, 256))
    glcm = graycomatrix(img_gray_resized, distances=[1], angles=[0], symmetric=True, normed=True)
    return {
        "skewness_R": img_skewness[0], "skewness_G": img_skewness[1], "skewness_B": img_skewness[2],
        "kurtosis_R": img_kurtosis[0], "kurtosis_G": img_kurtosis[1], "kurtosis_B": img_kurtosis[2],
        "entropy": img_entropy,
        "contrast": graycoprops(glcm, "contrast")[0, 0],
        "energy": graycoprops(glcm, "energy")[0, 0],
        "homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
        "rms_contrast": np.sqrt(np.mean((img_gray - np.mean(img_gray)) ** 2)),
        "sharpness": cv2.Laplacian(img_gray, cv2.CV_64F).var(),
    }


def process_image(image_path, write_image_stats, legacy_v1_metrics):
    img = None
    try:
        img = np.array(Image.open(image_path))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        sum_pixels, sum_squares, min_pixel, max_pixel, total_pixels = compute_intermediate_stats(img)
        image_stats = None
        if write_image_stats:
            image_stats = build_basic_image_stats(img)
            if legacy_v1_metrics:
                image_stats.update(build_legacy_v1_metrics(img))
        return sum_pixels, sum_squares, min_pixel, max_pixel, total_pixels, image_stats
    except Exception as exc:
        logging.error("Error processing %s: %s", image_path, exc)
        return None
    finally:
        del img
        gc.collect()


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    folder_name = os.path.basename(os.path.normpath(args.root_folder))
    log_file_path = os.path.join(args.output_folder, f"{folder_name}_processing.log")
    global_stats_csv_path = os.path.join(args.output_folder, f"{folder_name}_global_stats.csv")
    image_stats_csv_path = os.path.join(args.output_folder, f"{folder_name}_image_statistics.csv")
    setup_logging(log_file_path)
    if args.legacy_v1_metrics and not args.write_image_stats:
        logging.warning("--legacy-v1-metrics has no effect unless --write-image-stats is enabled.")
    global_sum_pixels = np.zeros(3, dtype=np.float64)
    global_sum_squares = np.zeros(3, dtype=np.float64)
    global_min_pixel = np.full(3, 255, dtype=np.float64)
    global_max_pixel = np.zeros(3, dtype=np.float64)
    global_total_pixels = 0
    image_rows = []
    files_to_process = []
    for subdir, _, files in os.walk(args.root_folder):
        for file in files:
            if file.lower().endswith(".png"):
                files_to_process.append(os.path.join(subdir, file))
    logging.info("Found %d images.", len(files_to_process))
    for image_path in tqdm(files_to_process, desc="Processing images", unit="image"):
        result = process_image(image_path, args.write_image_stats, args.legacy_v1_metrics)
        if result is None:
            continue
        sum_pixels, sum_squares, min_pixel, max_pixel, total_pixels, image_stats = result
        global_sum_pixels += sum_pixels
        global_sum_squares += sum_squares
        global_min_pixel = np.minimum(global_min_pixel, min_pixel)
        global_max_pixel = np.maximum(global_max_pixel, max_pixel)
        global_total_pixels += total_pixels
        if args.write_image_stats and image_stats is not None:
            image_rows.append({"filename": os.path.splitext(os.path.basename(image_path))[0], **image_stats})
    global_stats = {
        "global_sum_pixels_R": global_sum_pixels[0], "global_sum_pixels_G": global_sum_pixels[1], "global_sum_pixels_B": global_sum_pixels[2],
        "global_sum_squares_R": global_sum_squares[0], "global_sum_squares_G": global_sum_squares[1], "global_sum_squares_B": global_sum_squares[2],
        "global_min_pixel_R": global_min_pixel[0], "global_min_pixel_G": global_min_pixel[1], "global_min_pixel_B": global_min_pixel[2],
        "global_max_pixel_R": global_max_pixel[0], "global_max_pixel_G": global_max_pixel[1], "global_max_pixel_B": global_max_pixel[2],
        "global_total_pixels": global_total_pixels,
    }
    pd.DataFrame([global_stats]).to_csv(global_stats_csv_path, index=False)
    logging.info("Wrote global stats: %s", global_stats_csv_path)
    if args.write_image_stats:
        pd.DataFrame(image_rows).to_csv(image_stats_csv_path, index=False)
        logging.info("Wrote per-image stats: %s", image_stats_csv_path)


if __name__ == "__main__":
    main()
