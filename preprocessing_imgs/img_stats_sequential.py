import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
import cv2  # OpenCV for image resizing and Laplacian calculation
import logging
import gc

# ================================
# Configurable Parameters
# ================================

LOG_FILE_PATH = 'D:\\2024_PHD-DATA_CFP_PROCESSING\\2024-08-25_processing.log'
STATS_CSV_PATH = 'D:\\2024_PHD-DATA_CFP_PROCESSING\\2024-08-25_image_statistics_checkpoint.csv'
PROCESSED_IMAGES_PATH = 'D:\\2024_PHD-DATA_CFP_PROCESSING\\2024-08-25_processed_images.txt'
ROOT_FOLDER = r"D:\2024_PHD-DATA_CFP-imgs_cropped"
RESIZE_DIMENSIONS = (256, 256)
CHECKPOINT_FREQUENCY = 3000

# ================================
# Setup Logging
# ================================

logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE_PATH)
console_handler = logging.StreamHandler()

file_handler.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ================================
# Main Processing Function
# ================================

def process_image(subdir, file):
    filename = os.path.splitext(file)[0]
    img_path = os.path.join(subdir, file)

    img = None
    img_flat = None
    img_gray = None
    img_gray_resized = None
    glcm = None

    try:
        img = np.array(Image.open(img_path))

        if img.ndim == 2:  # Grayscale image
            img = np.stack([img] * 3, axis=-1)

        median = np.median(img, axis=(0, 1))
        min_pixel = img.min(axis=(0, 1))
        max_pixel = img.max(axis=(0, 1))

        img_flat = img.reshape(-1, 3)
        img_skewness = skew(img_flat, axis=0)
        img_kurtosis = kurtosis(img_flat, axis=0)
        img_entropy = shannon_entropy(img)

        img_gray = np.mean(img, axis=2).astype(np.uint8)
        img_gray_resized = cv2.resize(img_gray, RESIZE_DIMENSIONS)

        glcm = graycomatrix(img_gray_resized, distances=[1], angles=[0], symmetric=True, normed=True)
        img_contrast = graycoprops(glcm, 'contrast')[0, 0]
        img_energy = graycoprops(glcm, 'energy')[0, 0]
        img_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        rms_contrast = np.sqrt(np.mean((img_gray - np.mean(img_gray)) ** 2))
        sharpness = cv2.Laplacian(img_gray, cv2.CV_64F).var()

        return filename, {
            'median': median,
            'min_pixel': min_pixel,
            'max_pixel': max_pixel,
            'skewness': img_skewness,
            'kurtosis': img_kurtosis,
            'entropy': img_entropy,
            'contrast': img_contrast,
            'energy': img_energy,
            'homogeneity': img_homogeneity,
            'rms_contrast': rms_contrast,
            'sharpness': sharpness
        }, img.sum(axis=(0, 1)), (img ** 2).sum(axis=(0, 1)), img.shape[0] * img.shape[1]

    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")
    finally:
        del img, img_flat, img_gray, img_gray_resized, glcm
        gc.collect()
    return None


# ================================
# Main Program Execution
# ================================

def main():
    logging.info("Processing started.")

    if os.path.exists(STATS_CSV_PATH):
        stats_df = pd.read_csv(STATS_CSV_PATH, index_col='filename')
        processed_images = set(stats_df.index)
        logging.info(f"Loaded {len(processed_images)} previously processed images.")
    else:
        stats_df = pd.DataFrame(columns=[
            'median', 'min_pixel', 'max_pixel', 'skewness',
            'kurtosis', 'entropy', 'contrast', 'energy', 'homogeneity',
            'rms_contrast', 'sharpness'
        ])
        processed_images = set()
        logging.info("...no previous statistics found. Starting from scratch.")

    if os.path.exists(PROCESSED_IMAGES_PATH):
        with open(PROCESSED_IMAGES_PATH, 'r') as f:
            processed_images.update(f.read().splitlines())

    sum_pixels = np.zeros(3)
    sum_squared_pixels = np.zeros(3)
    total_pixels = 0

    total_images = sum(
        len(files) for _, _, files in os.walk(ROOT_FOLDER) if any(f.endswith('.png') for f in files)
    )

    first_1000_time_start = time.time()

    logging.info("...walking through images...")
    futures = []

    for subdir, _, files in os.walk(ROOT_FOLDER):
        for file in files:
            if file.endswith('.png'):
                filename = os.path.splitext(file)[0]
                if filename not in processed_images:
                    futures.append((subdir, file))

    logging.info(f"Processing {len(futures)} images...")

    for subdir, file in tqdm(futures, total=len(futures)):
        result = process_image(subdir, file)

        if result is None:
            logging.error(f"Processing failed for {file}.")
            continue

        filename, stats, img_sum, img_squared_sum, img_pixel_count = result
        sum_pixels += img_sum
        sum_squared_pixels += img_squared_sum
        total_pixels += img_pixel_count
        stats_df.loc[filename] = stats
        processed_images.add(filename)

        if len(processed_images) == CHECKPOINT_FREQUENCY:
            first_1000_time_end = time.time()
            time_per_image = (first_1000_time_end - first_1000_time_start) / CHECKPOINT_FREQUENCY
            remaining_images = total_images - len(processed_images)
            estimated_total_time = remaining_images * time_per_image
            logging.info(
                f"First {CHECKPOINT_FREQUENCY} imgs took: {first_1000_time_end - first_1000_time_start:.2f} seconds.")
            logging.info(f"Estimated remaining time: {estimated_total_time / 3600:.2f} hours.")
            stats_df.to_csv(STATS_CSV_PATH, index=True)
            with open(PROCESSED_IMAGES_PATH, 'w') as f:
                f.write("\n".join(processed_images))
            gc.collect()
            logging.info(f"Checkpoint saved after processing {len(processed_images)} images.")
        elif len(processed_images) % CHECKPOINT_FREQUENCY == 0:
            stats_df.to_csv(STATS_CSV_PATH, index=True)
            with open(PROCESSED_IMAGES_PATH, 'w') as f:
                f.write("\n".join(processed_images))
            gc.collect()
            logging.info(f"Checkpoint saved after processing {len(processed_images)} images.")
    logging.info("Storing stats")
    stats_df.to_csv(STATS_CSV_PATH, index=True)
    with open(PROCESSED_IMAGES_PATH, 'w') as f:
        f.write("\n".join(processed_images))
    logging.info("Final checkpoint saved.")

    logging.info("Calculating global statistics...")
    mean = sum_pixels / total_pixels
    variance = (sum_squared_pixels / total_pixels) - (mean ** 2)
    std_dev = np.sqrt(variance)
    print(f"Global Mean: {mean}")
    print(f"Global Standard Deviation: {std_dev}")
    logging.info("Processing completed.")


if __name__ == "__main__":
    main()
