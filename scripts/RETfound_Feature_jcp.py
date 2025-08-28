# scripts/RETFound_Feature.py

import os
import torch
import models_vit
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd


# 1. Load the model
def prepare_model(chkpt_dir, arch = 'vit_large_patch16'):
    model = models_vit.__dict__[arch](img_size = 224, num_classes = 0, drop_path_rate = 0.0, global_pool = True, )
    checkpoint = torch.load(chkpt_dir, map_location = 'cpu', weights_only = False)
    model.load_state_dict(checkpoint['model'], strict = False)
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
    model = torch.compile(model)  # JIT Compilation with torch.compile()
    print('Model loaded successfully.')

    # Define output directory
    output_dir = "E:\\PHD One Drive\\OneDrive - University of Edinburgh\\_PHD_DATA_derived\\0.2.UKB_CFP_derived\\CFP_RFpt\\"
    os.makedirs(output_dir, exist_ok = True)
    intermediate_dir = os.path.join(output_dir, "intermediate_files")
    os.makedirs(intermediate_dir, exist_ok = True)

    # 4. Get image folder paths - subfolders must be inside imgs
    data_root = 'B:\\2024_PHD-DATA_CFP-imgs_cropped\\'
    image_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    print(image_dirs)

    image_paths = []
    for dir_path in image_dirs:
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.png')]
        image_paths.extend(files)
    print(f'Total images found: {len(image_paths)}')

    # 5. DataLoader
    batch_size = 64  # Adjust as needed
    save_every_n_batches = 100  # Save every 100 batches
    dataset = FundusDataset(image_paths)
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 8,  # Adjust based on your CPU
        pin_memory = True
    )

    # 6. Feature extraction and saving intermediate files
    # data_rows = []
    chunk_data_rows = []
    chunk_counter = 0
    model.eval()

    for i, (batch_images, batch_names) in enumerate(tqdm(dataloader, desc = "Extracting features")):
        batch_images = batch_images.to(device)
        with torch.no_grad():
            batch_features = model.forward_features(batch_images)
        batch_features = batch_features.cpu().numpy()  # Shape: [batch_size, feature_dim]

        for name, feature in zip(batch_names, batch_features):
            row = {'image_name':name}
            row.update({f'feature_{idx}':val for idx, val in enumerate(feature)})
            chunk_data_rows.append(row)

        # Save a chunk of batches to a single intermediate file
        if (i + 1) % save_every_n_batches == 0:
            chunk_df = pd.DataFrame(chunk_data_rows)
            intermediate_file_path = os.path.join(intermediate_dir, f'chunk_{chunk_counter}.csv')
            chunk_df.to_csv(intermediate_file_path, index = False)
            chunk_counter += 1
            chunk_data_rows = []  # Reset for the next chunk

    # Save any remaining data that didn't form a full chunk
    if chunk_data_rows:
        chunk_df = pd.DataFrame(chunk_data_rows)
        intermediate_file_path = os.path.join(intermediate_dir, f'chunk_{chunk_counter}.csv')
        chunk_df.to_csv(intermediate_file_path, index = False)

    # 7. Combine intermediate files and save final Parquet file
    print("Combining intermediate files...")
    all_dfs = []
    intermediate_files = [os.path.join(intermediate_dir, f) for f in os.listdir(intermediate_dir) if f.endswith('.csv')]
    for f in tqdm(intermediate_files, desc = "Reading intermediate files"):
        try:
            all_dfs.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            print(f"Warning: Skipping empty intermediate file {f}")

    # 7. Save features
    if all_dfs:
        df = pd.concat(all_dfs, ignore_index = True)
        df.set_index('image_name', inplace = True)

        output_file = os.path.join(output_dir, 'CFP_RFpt.parquet')
        df.to_parquet(output_file, index = True, engine = 'pyarrow')
        print(f"Feature extraction and saving completed: {output_file}")
    else:
        print("No features were extracted. Final file not created.")
