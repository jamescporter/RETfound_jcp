import pandas as pd
import numpy as np
import glob

# Paths to the individual stats files
stats_files = glob.glob("D:\\2024_PHD-DATA_CFP_PROCESSING\\*image_statistics_checkpoint.csv")

all_stats = pd.concat([pd.read_csv(f, index_col='filename') for f in stats_files])

# Calculate the global statistics
sum_pixels = np.sum([all_stats['sum_pixels'].sum()])
sum_squared_pixels = np.sum([all_stats['sum_squared_pixels'].sum()])
total_pixels = all_stats['total_pixels'].sum()

global_mean = sum_pixels / total_pixels
global_variance = (sum_squared_pixels / total_pixels) - (global_mean ** 2)
global_std_dev = np.sqrt(global_variance)

print(f"Global Mean: {global_mean}")
print(f"Global Standard Deviation: {global_std_dev}")

# Save the aggregated stats if needed
all_stats.to_csv('D:\\2024_PHD-DATA_CFP_PROCESSING\\final_image_statistics.csv')
