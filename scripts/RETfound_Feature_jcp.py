# scripts/RETFound_Feature.py

import os
import torch
import models_vit
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from functools import partial
import torch.nn as nn
from tqdm import tqdm
import h5py
import pandas as pd  # Added import


# 1. Load the model
def prepare_model(chkpt_dir, arch='vit_large_patch16'):
    model = models_vit.__dict__[arch](
        img_size=224,
        num_classes=0,
        drop_path_rate=0.0,
        global_pool=True,
    )
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    return model


# 2. Image preprocessing
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = (img - imagenet_mean) / imagenet_std
    img = torch.tensor(img).permute(2, 0, 1)  # Shape: [3, 224, 224]
    return img.float()


# my_mean = np.array([73.9277, 37.6650, 13.7630])
# my_std = np.array([88.0669, 45.9370, 20.4242])
#
#
# def preprocess_image(image_path):
#     img = Image.open(image_path).convert('RGB')
#     img = img.resize((224, 224))
#     img = np.array(img) / 255.0
#     img = (img - my_mean) / my_std
#     img = torch.tensor(img).permute(2, 0, 1)  # Shape: [3, 224, 224]
#     return img.float()


# 3. Dataset and DataLoader
class FundusDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_tensor = preprocess_image(self.image_paths[idx])
        img_name = os.path.basename(self.image_paths[idx])
        return img_tensor, img_name


# Main execution block
if __name__ == '__main__':
    # 1. Load the model
    chkpt_dir = '../models/Model_checkpoints/RETFound_cfp_weights.pth'  # Adjust path as needed
    model = prepare_model(chkpt_dir, 'vit_large_patch16')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Model loaded successfully.')

    # 4. Get image folder paths - subfolders must be inside imgs
    data_root = 'B:\\2024_PHD-DATA_CFP-imgs_cropped\\'
    image_dirs = [
        os.path.join(data_root, d) for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ]

    print(image_dirs)

    image_paths = []
    for dir_path in image_dirs:
        files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith('.png')
        ]
        image_paths.extend(files)

    print(f'Total images found: {len(image_paths)}')

    # 5. DataLoader
    batch_size = 64  # Adjust as needed
    dataset = FundusDataset(image_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Adjust based on your CPU
        pin_memory=True
    )

    # 6. Feature extraction
    data_rows = []

    model.eval()
    for batch_images, batch_names in tqdm(dataloader):
        batch_images = batch_images.to(device)
        with torch.no_grad():
            batch_features = model.forward_features(batch_images)
        batch_features = batch_features.cpu().numpy()  # Shape: [batch_size, feature_dim]

        for name, feature in zip(batch_names, batch_features):
            row = {'image_name': name}
            for idx, value in enumerate(feature):
                row[f'feature_{idx}'] = value
            data_rows.append(row)

    # 7. Save features to CSV
    output_dir = "E:\\PHD One Drive\\OneDrive - University of Edinburgh\\PHD_OUTPUT\\2024-10 RETfound features\output_imagenet\\"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'features.csv')

    # Create DataFrame
    df = pd.DataFrame(data_rows)
    df.set_index('image_name', inplace=True)

    # Save to CSV
    df.to_csv(output_file)
    print("Feature extraction and saving to CSV completed.")
