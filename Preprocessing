#
import cv2  # OpenCV for image processing
import numpy as np  # NumPy for array manipulations
import os  # Operating system utilities for file handling
import gdown  # Utility to download files from Google Drive

# Step 1: Function to download the dataset from Google Drive using gdown
def download_dataset():
    """
    Downloads the dataset from a public Google Drive folder link using gdown.
    Adjust this function if you need to handle specific files instead of folders.
    """
    google_drive_folder_url = 'Path in dataset'
    
    # gdown.download_folder will download the entire folder and save it locally
    print("Downloading dataset from Google Drive...")
    gdown.download_folder(google_drive_folder_url, quiet=False)
    print("Download completed!")

# Step 2: Function to apply preprocessing on MRI images
def preprocess_image(image):
    """
    Applies the following preprocessing steps on an input image:
    1. Gaussian filtering for noise removal.
    2. CLAHE for contrast enhancement.
    3. Resizing the image to 128x128x3 using bilinear interpolation.
    
    Args:
        image (numpy array): The input MRI image to preprocess.
    
    Returns:
        numpy array: The preprocessed image.
    """
    
    # Gaussian filter to smooth the image and remove noise
    # Kernel size: 3x3, Standard deviation (sigma): 1.5
    print("Applying Gaussian filter...")
    gaussian_filtered = cv2.GaussianBlur(image, (3, 3), 1.5)
    
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # Clip limit: 2.0, Tile grid size: 8x8
    print("Applying CLAHE for contrast enhancement...")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    if len(image.shape) == 3:  # If the image is colored (3 channels, RGB/BGR)
        # Convert the image to LAB color space to apply CLAHE only on the L channel (luminance)
        lab = cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)  # Splitting LAB channels
        l_clahe = clahe.apply(l)  # Apply CLAHE on the luminance channel
        lab_clahe = cv2.merge((l_clahe, a, b))  # Merge channels back
        image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)  # Convert back to BGR/RGB
    else:  # If the image is grayscale (1 channel)
        image_clahe = clahe.apply(gaussian_filtered)  # Directly apply CLAHE on the image
    
    # Resize the image to 128x128x3 using bilinear interpolation
    print("Resizing the image to 128x128...")
    resized_image = cv2.resize(image_clahe, (128, 128), interpolation=cv2.INTER_LINEAR)
    
    # Return the preprocessed image
    return resized_image

# Step 3: Function to load and preprocess all images from a folder
def process_images_from_folder(folder_path):
    """
    Iterates through a folder containing MRI images, preprocesses each image, and stores the preprocessed images.
    
    Args:
        folder_path (str): The path to the folder containing the images.
    
    Returns:
        list: A list containing all the preprocessed images.
    """
    
    processed_images = []  # List to hold preprocessed images
    
    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        # Only process files that are in image formats (e.g., .png, .jpg)
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)  # Full path to the image file
            
            # Load the image using OpenCV
            print(f"Loading image: {filename}")
            image = cv2.imread(image_path)
            
            if image is not None:
                # Apply the preprocessing steps to the loaded image
                processed_image = preprocess_image(image)
                
                # Add the preprocessed image to the list
                processed_images.append(processed_image)
                
                # Optionally, save the preprocessed image (can be removed if not needed)
                preprocessed_image_path = os.path.join(folder_path, f"processed_{filename}")
                cv2.imwrite(preprocessed_image_path, processed_image)
                print(f"Saved preprocessed image as: processed_{filename}")
    
    # Return the list of preprocessed images
    return processed_images

# Step 4: Main function to execute the preprocessing pipeline
def main():
    # Step 1: Download the dataset from Google Drive
    download_dataset()
    
    # Step 2: Specify the path to the downloaded dataset folder
    # Update this path with the actual folder name where your dataset is saved after downloading
    dataset_folder_path = './your_downloaded_dataset_folder'  # Replace with the actual folder name
    
    # Step 3: Preprocess all images from the folder
    print("Starting image preprocessing...")
    processed_images = process_images_from_folder(dataset_folder_path)
    
    # Print the shape of the first preprocessed image as an example
    if processed_images:
        print("Preprocessing completed. Example of preprocessed image shape:", processed_images[0].shape)

# Run the main function to start the preprocessing
if __name__ == "__main__":
    main()
