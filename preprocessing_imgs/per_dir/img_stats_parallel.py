import os
import sys
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

# Expect the folder path as a command-line argument
if len(sys.argv) < 2:
    print("Please provide the folder path as a command-line argument.")
    sys.exit(1)

ROOT_FOLDER = sys.argv[1]

# Define the output directory
OUTPUT_FOLDER = r"D:\2024_PHD-DATA_CFP_PROCESSING_v2"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Generate unique names for the log file and the stats CSV file based on the input folder name
folder_name = os.path.basename(os.path.normpath(ROOT_FOLDER))
LOG_FILE_PATH = os.path.join(OUTPUT_FOLDER, f'{folder_name}_processing.log')
STATS_CSV_PATH = os.path.join(OUTPUT_FOLDER, f'{folder_name}_image_statistics.csv')

RESIZE_DIMENSIONS = (256, 256)

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
        }

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

    stats_df = pd.DataFrame(columns=[
        'median', 'min_pixel', 'max_pixel', 'skewness',
        'kurtosis', 'entropy', 'contrast', 'energy', 'homogeneity',
        'rms_contrast', 'sharpness'
    ])

    logging.info("...walking through images...")
    futures = []

    for subdir, _, files in os.walk(ROOT_FOLDER):
        for file in files:
            if file.endswith('.png'):
                futures.append((subdir, file))

    logging.info(f"Processing {len(futures)} images...")

    for subdir, file in tqdm(futures, total=len(futures)):
        result = process_image(subdir, file)

        if result is None:
            logging.error(f"Processing failed for {file}.")
            continue

        filename, stats = result
        stats_df.loc[filename] = stats

    logging.info("Storing stats")
    stats_df.to_csv(STATS_CSV_PATH, index=True)
    logging.info("Processing completed.")


if __name__ == "__main__":
    main()
