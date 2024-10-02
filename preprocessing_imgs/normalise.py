import os
from PIL import Image
from torchvision import transforms

# Define your custom mean and standard deviation
custom_mean = [0.5, 0.5, 0.5]  # Replace with your calculated values
custom_std = [0.2, 0.2, 0.2]  # Replace with your calculated values

# Define the preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize(224),  # Resize the shorter side to 224 pixels
    transforms.CenterCrop(224),  # Crop the central 224x224 pixels
    transforms.ToTensor(),
    transforms.Normalize(mean=custom_mean, std=custom_std),  # Apply custom normalization
])

# Directory paths
input_folder = "path_to_your_input_folder"
output_folder = "path_to_your_output_folder"

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):  # Only process PNG images
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert("RGB")  # Open and ensure image is in RGB mode

        # Apply the preprocessing transformations
        image_tensor = preprocess(image)

        # Convert the tensor back to a PIL image for saving (unnormalize first)
        unnormalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(custom_mean, custom_std)],
            std=[1 / s for s in custom_std]
        )
        image_tensor = unnormalize(image_tensor).clamp(0, 1)  # Clamp to ensure valid pixel range
        processed_image = transforms.ToPILImage()(image_tensor)

        # Save the processed image
        output_path = os.path.join(output_folder, filename)
        processed_image.save(output_path)

print("Processing complete. Images saved to:", output_folder)
