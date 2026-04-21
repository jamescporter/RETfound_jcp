import os
import numpy as np
import cv2
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from tqdm import tqdm


def calculate_image_statistics(image_folder):
    # List all image files
    image_files = [entry.path for entry in os.scandir(image_folder) if entry.name.endswith(".png")]
    n_files = len(image_files)

    means = np.zeros(n_files, dtype=np.float32)
    stds = np.zeros(n_files, dtype=np.float32)
    histograms = np.zeros((n_files, 256), dtype=np.float32)
    skewnesses = np.zeros(n_files, dtype=np.float32)
    kurtoses = np.zeros(n_files, dtype=np.float32)
    contrasts = np.zeros(n_files, dtype=np.float32)
    entropies = np.zeros(n_files, dtype=np.float32)
    dynamic_ranges = np.zeros(n_files, dtype=np.float32)

    for idx, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        # Calculate mean and std
        means[idx] = np.mean(image)
        stds[idx] = np.std(image)

        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        histograms[idx] = hist

        # Flatten the image for further calculations
        image_flatten = image.flatten()

        # Calculate skewness
        skewnesses[idx] = skew(image_flatten)

        # Calculate kurtosis
        kurtoses[idx] = kurtosis(image_flatten)

        # Calculate contrast
        contrasts[idx] = image.max() - image.min()

        # Calculate entropy
        entropies[idx] = shannon_entropy(image)

        # Calculate dynamic range
        dynamic_ranges[idx] = np.ptp(image)

    # Calculate overall statistics
    overall_mean = np.mean(means)
    overall_std = np.mean(stds)
    overall_histogram = np.mean(histograms, axis=0)
    overall_skewness = np.mean(skewnesses)
    overall_kurtosis = np.mean(kurtoses)
    overall_contrast = np.mean(contrasts)
    overall_entropy = np.mean(entropies)
    overall_dynamic_range = np.mean(dynamic_ranges)

    return {
        "mean": overall_mean,
        "std": overall_std,
        "histogram": overall_histogram,
        "skewness": overall_skewness,
        "kurtosis": overall_kurtosis,
        "contrast": overall_contrast,
        "entropy": overall_entropy,
        "dynamic_range": overall_dynamic_range
    }


image_folder = r"B:\2024_PHD-DATA_CFP-imgs_cropped\1-20k_240128-2147"
statistics = calculate_image_statistics(image_folder)
print(statistics)
