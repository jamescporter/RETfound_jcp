import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

# Load all global stats files
stats_files = glob.glob("D:\\2024_PHD-DATA_CFP_PROCESSING_v2\\*_global_stats.csv")

# Initialize global accumulators for each channel (R, G, B)
global_sum_pixels = np.zeros(3, dtype=np.float64)
global_sum_squares = np.zeros(3, dtype=np.float64)
global_min_pixel = np.full(3, 255, dtype=np.float64)
global_max_pixel = np.zeros(3, dtype=np.float64)
global_total_pixels = 0

# Process each stats file with a progress bar
for file in tqdm(stats_files, desc="Combining Global Stats", unit="file"):
    stats = pd.read_csv(file)

    global_sum_pixels += np.array([
        stats.iloc[0]['global_sum_pixels_R'],
        stats.iloc[0]['global_sum_pixels_G'],
        stats.iloc[0]['global_sum_pixels_B']
    ])

    global_sum_squares += np.array([
        stats.iloc[0]['global_sum_squares_R'],
        stats.iloc[0]['global_sum_squares_G'],
        stats.iloc[0]['global_sum_squares_B']
    ])

    global_min_pixel = np.minimum(global_min_pixel, np.array([
        stats.iloc[0]['global_min_pixel_R'],
        stats.iloc[0]['global_min_pixel_G'],
        stats.iloc[0]['global_min_pixel_B']
    ]))

    global_max_pixel = np.maximum(global_max_pixel, np.array([
        stats.iloc[0]['global_max_pixel_R'],
        stats.iloc[0]['global_max_pixel_G'],
        stats.iloc[0]['global_max_pixel_B']
    ]))

    global_total_pixels += stats.iloc[0]['global_total_pixels']


# Calculate the final global statistics for each channel
def compute_final_stats(sum_pixels, sum_squares, total_pixels):
    global_mean = sum_pixels / total_pixels
    global_variance = (sum_squares / total_pixels) - (global_mean ** 2)
    global_std_dev = np.sqrt(global_variance)
    return global_mean, global_variance, global_std_dev


global_mean, global_variance, global_std_dev = compute_final_stats(global_sum_pixels, global_sum_squares,
                                                                   global_total_pixels)

# Combine results into a DataFrame
global_stats = pd.DataFrame({
    'global_mean_R': [global_mean[0]],
    'global_mean_G': [global_mean[1]],
    'global_mean_B': [global_mean[2]],
    'global_variance_R': [global_variance[0]],
    'global_variance_G': [global_variance[1]],
    'global_variance_B': [global_variance[2]],
    'global_std_dev_R': [global_std_dev[0]],
    'global_std_dev_G': [global_std_dev[1]],
    'global_std_dev_B': [global_std_dev[2]],
    'global_min_pixel_R': [global_min_pixel[0]],
    'global_min_pixel_G': [global_min_pixel[1]],
    'global_min_pixel_B': [global_min_pixel[2]],
    'global_max_pixel_R': [global_max_pixel[0]],
    'global_max_pixel_G': [global_max_pixel[1]],
    'global_max_pixel_B': [global_max_pixel[2]],
    'global_total_pixels': [global_total_pixels]
})

# Print the final global statistics
print(global_stats)
global_stats.to_csv('D:\\2024_PHD-DATA_CFP_PROCESSING_v2\\final_full_stats.csv')

# Combine the old statistics with a progress bar
old_stats_files = glob.glob("D:\\2024_PHD-DATA_CFP_PROCESSING\\*image_statistics_checkpoint.csv")

# Check the number of rows in each file and the shape of each DataFrame
total_rows = 0
for f in old_stats_files:
    df = pd.read_csv(f, index_col=0)
    print(f"{f}: Shape = {df.shape}")
    total_rows += len(df)

print(f"Expected total rows (9 files * ~19k per file): {len(old_stats_files) * 19000}")
print(f"Actual total rows: {total_rows}")

# Concatenate and log number of rows in the combined DataFrame
all_old_stats = pd.concat(
    [pd.read_csv(f, index_col=0) for f in tqdm(old_stats_files, desc="Combining Old Stats", unit="file")])
print(f"Combined DataFrame rows: {len(all_old_stats)}")

# Split multi-value columns into separate columns
for column in tqdm(['median', 'min_pixel', 'max_pixel', 'skewness', 'kurtosis'], desc="Splitting Columns",
                   unit="column"):
    all_old_stats[[f"{column}_R", f"{column}_G", f"{column}_B"]] = all_old_stats[column].str.strip('[]').str.split(
        expand=True).astype(float)
    all_old_stats.drop(columns=[column], inplace=True)

# Save the combined results
all_old_stats.to_csv('D:\\2024_PHD-DATA_CFP_PROCESSING_v2\\final_image_statistics.csv')
