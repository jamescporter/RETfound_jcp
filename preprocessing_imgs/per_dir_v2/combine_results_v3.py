import pandas as pd
import numpy as np
from tqdm import tqdm

# List of all the specified files
old_stats_files = [
    "D:\\2024_PHD-DATA_CFP_PROCESSING_v2\\161-181k_240130-0031_image_statistics.csv",
    "D:\\2024_PHD-DATA_CFP_PROCESSING_v2\\1-20k_240128-2147_image_statistics.csv",
    "D:\\2024_PHD-DATA_CFP_PROCESSING_v2\\21-40k_240128-2211_image_statistics.csv",
    "D:\\2024_PHD-DATA_CFP_PROCESSING_v2\\41-60k_240128-2240_image_statistics.csv",
    "D:\\2024_PHD-DATA_CFP_PROCESSING_v2\\61-80k_240128-2307_image_statistics.csv",
    "D:\\2024_PHD-DATA_CFP_PROCESSING_v2\\81-100k_240128-2346_image_statistics.csv",
    "D:\\2024_PHD-DATA_CFP_PROCESSING_v2\\101-120k_240129-1215_image_statistics.csv",
    "D:\\2024_PHD-DATA_CFP_PROCESSING_v2\\121-140k_240129-2346_image_statistics.csv",
    "D:\\2024_PHD-DATA_CFP_PROCESSING_v2\\141-160k_240130-0008_image_statistics.csv"
]

# Initialize the total row count
total_rows = 0

# Process each file, print its shape, and accumulate the total number of rows
for f in old_stats_files:
    df = pd.read_csv(f, index_col=0)
    print(f"{f}: Shape = {df.shape}")
    total_rows += len(df)

print(f"Actual total rows: {total_rows}")

# Concatenate all the DataFrames
all_old_stats = pd.concat(
    [pd.read_csv(f, index_col=0) for f in tqdm(old_stats_files, desc="Combining Old Stats", unit="file")])

# Print the shape of the combined DataFrame
print(f"Combined DataFrame rows: {len(all_old_stats)}")

# Split multi-value columns into separate columns
for column in tqdm(['median', 'min_pixel', 'max_pixel', 'skewness', 'kurtosis'], desc="Splitting Columns",
                   unit="column"):
    all_old_stats[[f"{column}_R", f"{column}_G", f"{column}_B"]] = all_old_stats[column].str.strip('[]').str.split(
        expand=True).astype(float)
    all_old_stats.drop(columns=[column], inplace=True)

# Save the combined results
all_old_stats.to_csv('D:\\2024_PHD-DATA_CFP_PROCESSING_v2\\final_image_statistics.csv')
