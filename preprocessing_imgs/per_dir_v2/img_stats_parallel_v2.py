import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import logging
import gc
from tqdm import tqdm
from numba import njit, prange

# Configurable Parameters
if len(sys.argv) < 2:
    print("Please provide the folder path as a command-line argument.")
    sys.exit(1)

ROOT_FOLDER = sys.argv[1]

OUTPUT_FOLDER = r"D:\2024_PHD-DATA_CFP_PROCESSING_v2"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

folder_name = os.path.basename(os.path.normpath(ROOT_FOLDER))
LOG_FILE_PATH = os.path.join(OUTPUT_FOLDER, f'{folder_name}_processing.log')
STATS_CSV_PATH = os.path.join(OUTPUT_FOLDER, f'{folder_name}_global_stats.csv')

# Setup Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE_PATH)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


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


def process_image(subdir, file):
    img_path = os.path.join(subdir, file)
    try:
        img = np.array(Image.open(img_path))
        if img.ndim == 2:  # Grayscale image
            img = np.stack([img] * 3, axis=-1)

        sum_pixels, sum_squares, min_pixel, max_pixel, total_pixels = compute_intermediate_stats(img)

        return sum_pixels, sum_squares, min_pixel, max_pixel, total_pixels

    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")
        return None
    finally:
        del img
        gc.collect()


def main():
    logging.info("Processing started.")

    global_sum_pixels = np.zeros(3, dtype=np.float64)
    global_sum_squares = np.zeros(3, dtype=np.float64)
    global_min_pixel = np.full(3, 255, dtype=np.float64)
    global_max_pixel = np.zeros(3, dtype=np.float64)
    global_total_pixels = 0

    # Gather all files first to use with tqdm
    files_to_process = []
    for subdir, _, files in os.walk(ROOT_FOLDER):
        for file in files:
            if file.endswith('.png'):
                files_to_process.append((subdir, file))

    # Progress bar for processing images
    for subdir, file in tqdm(files_to_process, desc="Processing Images", unit="image"):
        result = process_image(subdir, file)

        if result is None:
            logging.error(f"Processing failed for {file}.")
            continue

        sum_pixels, sum_squares, min_pixel, max_pixel, total_pixels = result

        global_sum_pixels += sum_pixels
        global_sum_squares += sum_squares
        global_min_pixel = np.minimum(global_min_pixel, min_pixel)
        global_max_pixel = np.maximum(global_max_pixel, max_pixel)
        global_total_pixels += total_pixels

    # Combine results into a dictionary with arrays split across keys
    global_stats = {
        'global_sum_pixels_R': global_sum_pixels[0],
        'global_sum_pixels_G': global_sum_pixels[1],
        'global_sum_pixels_B': global_sum_pixels[2],
        'global_sum_squares_R': global_sum_squares[0],
        'global_sum_squares_G': global_sum_squares[1],
        'global_sum_squares_B': global_sum_squares[2],
        'global_min_pixel_R': global_min_pixel[0],
        'global_min_pixel_G': global_min_pixel[1],
        'global_min_pixel_B': global_min_pixel[2],
        'global_max_pixel_R': global_max_pixel[0],
        'global_max_pixel_G': global_max_pixel[1],
        'global_max_pixel_B': global_max_pixel[2],
        'global_total_pixels': global_total_pixels
    }

    logging.info("Storing global stats.")
    pd.DataFrame([global_stats]).to_csv(STATS_CSV_PATH, index=False)
    logging.info("Processing completed.")


if __name__ == "__main__":
    main()
