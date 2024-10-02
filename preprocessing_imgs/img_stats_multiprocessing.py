import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
from multiprocessing import Manager, cpu_count
import cv2  # OpenCV for image resizing and Laplacian calculation
import logging
import gc

# ================================
# Configurable Parameters
# ================================

# Path to save the log file
LOG_FILE_PATH = 'D:\\2024_PHD-DATA_CFP_PROCESSING\\2024-08-25_processing.log'

# Paths to checkpoint files
STATS_CSV_PATH = 'D:\\2024_PHD-DATA_CFP_PROCESSING\\2024-08-25_image_statistics_checkpoint.csv'
PROCESSED_IMAGES_PATH = 'D:\\2024_PHD-DATA_CFP_PROCESSING\\2024-08-25_processed_images.txt'

# Root folder containing all images
ROOT_FOLDER = r"D:\2024_PHD-DATA_CFP-imgs_cropped"

# Number of workers (all cores but 2)
NUM_WORKERS = 1  # max(cpu_count() - 4, 1)

# Image resizing parameters (for GLCM calculations)
RESIZE_DIMENSIONS = (512, 512)

# Frequency of checkpoint saves (every N images)
CHECKPOINT_FREQUENCY = 5000

# ================================
# Setup Logging
# ================================

# logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler(LOG_FILE_PATH)
console_handler = logging.StreamHandler()

# Set level for handlers
file_handler.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ================================
# Main Processing Function
# ================================

def process_image(subdir, file):
    try:
        filename = os.path.splitext(file)[0]
        img_path = os.path.join(subdir, file)

        # Simulate processing with a simple operation
        time.sleep(0.1)  # Simulate a small delay

        return filename, {
            'median': 1,
            'min_pixel': 1,
            'max_pixel': 1,
            'skewness': 1,
            'kurtosis': 1,
            'entropy': 1,
            'contrast': 1,
            'energy': 1,
            'homogeneity': 1,
            'rms_contrast': 1,
            'sharpness': 1
        }, np.array([1, 1, 1]), np.array([1, 1, 1]), 1

    except Exception as e:
        logging.error(f"Error processing image {img_path}: {e}")
        return None

    finally:
        gc.collect()


# def process_image(subdir, file):
#     filename = os.path.splitext(file)[0]
#     img_path = os.path.join(subdir, file)
#
#     # Initialize variables
#     img = None
#     img_flat = None
#     img_gray = None
#     img_gray_resized = None
#     glcm = None
#
#     try:
#         # Load image
#         img = np.array(Image.open(img_path))
#         #logging.info(f"Image opened: {img_path}")
#
#         # if img.ndim == 2:  # Grayscale image
#         #     img = np.stack([img] * 3, axis=-1)  # Convert to 3-channel grayscale
#         #     logging.info(f"{img_path} = Converted grayscale to 3-channel: {img.shape}")
#
#         # Calculate individual statistics
#         median = 1#np.median(img, axis=(0, 1))
#         min_pixel = 1#img.min(axis=(0, 1))
#         max_pixel = 1#img.max(axis=(0, 1))
#         #logging.info(f"{img_path} = Calculated basic stats: median={median}, min={min_pixel}, max={max_pixel}")
#
#         img_flat = img.reshape(-1, 3)
#         img_skewness = 1#skew(img_flat, axis=0)
#         img_kurtosis = 1#kurtosis(img_flat, axis=0)
#         img_entropy = 1#shannon_entropy(img)
#         #logging.info(f"{img_path} = Calculated skewness, kurtosis, and entropy.")
#
#         # Convert to grayscale for GLCM and additional statistics
#         img_gray = np.mean(img, axis=2).astype(np.uint8)
#         img_gray_resized = cv2.resize(img_gray, RESIZE_DIMENSIONS)
#         #logging.info(f"{img_path} = Resized grayscale image for GLCM calculation.")
#
#         # GLCM calculations
#         # glcm = graycomatrix(img_gray_resized, distances=[1], angles=[0], symmetric=True, normed=True)
#         # img_contrast = graycoprops(glcm, 'contrast')[0, 0]
#         # img_energy = graycoprops(glcm, 'energy')[0, 0]
#         # img_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
#         glcm = 1
#         img_contrast = 1
#         img_energy = 1
#         img_homogeneity = 1
#
#         #logging.info(f"{img_path} = Calculated GLCM properties.")
#
#         # Calculate RMS contrast
#         rms_contrast = np.sqrt(np.mean((img_gray - np.mean(img_gray)) ** 2))
#         #logging.info(f"{img_path} = Calculated RMS contrast.")
#
#         # Calculate image sharpness using the variance of the Laplacian
#         sharpness = cv2.Laplacian(img_gray, cv2.CV_64F).var()
#         #logging.info(f"{img_path} = Calculated image sharpness.")
#
#         # Clean up memory
#         #del img, img_flat, img_gray, img_gray_resized, glcm
#         #del img_flat, img_gray, img_gray_resized, glcm
#
#         # Return the filename and statistics for this image
#         return filename, {
#             'median': median,
#             'min_pixel': min_pixel,
#             'max_pixel': max_pixel,
#             'skewness': img_skewness,
#             'kurtosis': img_kurtosis,
#             'entropy': img_entropy,
#             'contrast': img_contrast,
#             'energy': img_energy,
#             'homogeneity': img_homogeneity,
#             'rms_contrast': rms_contrast,
#             'sharpness': sharpness
#         }, img.sum(axis=(0, 1)), (img ** 2).sum(axis=(0, 1)), img.shape[0] * img.shape[1]
#
#     except FileNotFoundError as e:
#         logging.error(f"File not found {img_path}: {e}")
#     except IOError as e:
#         logging.error(f"IOError for file {img_path}: {e}")
#     except AttributeError as e:
#         logging.error(f"Attribute error in {img_path} with {e}")
#     except Exception as e:
#         logging.error(f"General error processing {img_path}: {e}")
#     finally:
#         # Ensure that all variables are deleted to free memory
#         if img is not None:
#             del img
#         if img_flat is not None:
#             del img_flat
#         if img_gray is not None:
#             del img_gray
#         if img_gray_resized is not None:
#             del img_gray_resized
#         if glcm is not None:
#             del glcm
#         gc.collect()
#     #print(f"Image processed: {img_path}")
#     return None


# ================================
# Main Program Execution
# ================================

def main():
    logging.info("Processing started.")
    # Load previously processed data if exists
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

    # Load list of processed images if exists
    if os.path.exists(PROCESSED_IMAGES_PATH):
        with open(PROCESSED_IMAGES_PATH, 'r') as f:
            processed_images.update(f.read().splitlines())

    # Initialize accumulators for global mean and std dev
    sum_pixels = np.zeros(3)  # Assuming RGB images
    sum_squared_pixels = np.zeros(3)
    total_pixels = 0

    # Use a Manager to share the processed images set between processes
    with Manager() as manager:
        processed_images_dict = manager.dict({img: None for img in processed_images})
        processed_count = manager.Value('i', 0)
        # lock = manager.Lock()

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            total_images = sum(
                len(files) for _, _, files in os.walk(ROOT_FOLDER) if any(f.endswith('.png') for f in files))
            # logging.info(f"...images to process: {total_images}")
            first_1000_time_start = time.time()

            logging.info("...walking through images...")
            for subdir, _, files in os.walk(ROOT_FOLDER):
                for file in files:
                    if file.endswith('.png'):
                        filename = os.path.splitext(file)[0]
                        if filename not in processed_images_dict:
                            # futures.append(executor.submit(process_image, subdir, file, processed_count, lock))
                            futures.append(executor.submit(process_image, subdir, file))
            logging.info(f"Processing {len(futures)} images...")

            ##########################################

            for future in tqdm(as_completed(futures), total=len(futures)):
                logging.info(f"Beginning processing futures")
                try:
                    result = future.result()
                except Exception as e:
                    logging.error(f"Error processing future: {e}")
                    continue
                if result is None:
                    logging.error("Future is none.")
                    continue
                logging.info("Now processing results")

                filename, stats, img_sum, img_squared_sum, img_pixel_count = result
                sum_pixels += img_sum
                sum_squared_pixels += img_squared_sum
                total_pixels += img_pixel_count
                stats_df.loc[filename] = stats
                processed_images_dict[filename] = None

                # Increment the processed_count after successfully processing an image
                with processed_count.get_lock():
                    processed_count.value += 1

                if processed_count.value == CHECKPOINT_FREQUENCY:
                    first_1000_time_end = time.time()
                    time_per_image = (first_1000_time_end - first_1000_time_start) / CHECKPOINT_FREQUENCY
                    remaining_images = total_images - processed_count.value
                    estimated_total_time = remaining_images * time_per_image
                    logging.info(
                        f"First {CHECKPOINT_FREQUENCY} imgs took: {first_1000_time_end - first_1000_time_start:.2f} seconds.")
                    logging.info(f"Estimated remaining time: {estimated_total_time / 3600:.2f} hours.")

                if processed_count.value % CHECKPOINT_FREQUENCY == 0:
                    stats_df.to_csv(STATS_CSV_PATH)
                    with open(PROCESSED_IMAGES_PATH, 'w') as f:
                        f.write("\n".join(list(processed_images_dict.keys())))
                    gc.collect()
                    logging.info(f"Checkpoint saved after processing {processed_count.value} images.")

        logging.info("Storing stats")
        stats_df.to_csv(STATS_CSV_PATH, index=True)
        with open(PROCESSED_IMAGES_PATH, 'w') as f:
            f.write("\n".join(list(processed_images_dict.keys())))
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
