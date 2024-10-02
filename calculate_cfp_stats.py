import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def calculate_image_statistics(image_folder):
    means = []
    stds = []
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]

    for image_file in tqdm(image_files):
        image = Image.open(image_file).convert('RGB')
        image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
        means.append(image.mean(axis=(0, 1)))
        stds.append(image.std(axis=(0, 1)))

    # Compute global mean and std
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return mean, std


image_folder = '/path/to/your/images'  # Update with your actual image folder path
mean, std = calculate_image_statistics(image_folder)
print('Mean:', mean)
print('Standard Deviation:', std)
