# Mod-R2AU-Net for Brain Tumor Segmentation
Mod-R2AU-Net is an optimized and enhanced version of the U-Net architecture, specifically designed for advanced medical image segmentation, with a focus on brain tumor segmentation in MRI images. The model integrates Recurrent Residual Convolutional Layers (R2CL), Attention Gates (AGs), and Batch Normalization (BN) to significantly improve feature extraction, processing speed, and segmentation accuracy.

The R2CL blocks are employed in both the encoder and decoder paths, utilizing residual connections and recurrent learning to refine feature representation over multiple iterations. This allows the model to capture complex patterns in MRI images more effectively. Attention Gates (AGs) are applied to further enhance the segmentation process by focusing on the most relevant regions within the image, improving the model’s ability to distinguish between healthy and tumor tissues.

Batch Normalization (BN) is incorporated to stabilize and accelerate the training process, ensuring that each layer’s input is appropriately scaled and distributed. This results in faster convergence and improved generalization, making the model more robust in various segmentation tasks.

Despite its advanced architecture, Mod-R2AU-Net remains computationally efficient, reducing the number of parameters and computational demands while delivering high accuracy. With 212 GFlops and 21.3M parameters, it is a lightweight model suitable for deployment in real-world clinical settings. These optimizations allow for faster processing times and resource-friendly deployment on modern hardware, such as GPUs, making the model ideal for large-scale MRI datasets. Mod-R2AU-Net’s combination of R2CL, AGs, and BN ensures a strong balance between computational efficiency and precise brain tumor segmentation, making it a powerful tool for medical professionals and researchers.
## Proposed Architecture of Mod-R2AU-Net 

The graphical abstract illustrates the overall architecture of the Mod-R2AU-Net model, designed specifically for brain tumor segmentation using MRI images. This architecture enhances traditional U-Net by integrating Recurrent Residual Convolutional Layers (R2CL) and Attention Gates (AGs) to improve segmentation accuracy and model efficiency. Below is a detailed breakdown of the key components depicted:

### Input Layer

- The input to the model consists of MRI images of brain scans, resized to 128x128x3 dimensions.

### Downsampling Path (Encoder)

- The encoder path uses multiple layers of R2CL blocks. Each R2CL block iteratively processes the input to capture complex features and spatial hierarchies while preserving contextual information. This is achieved through residual learning, where feature maps from earlier layers are re-used, preventing the vanishing gradient problem.
- MaxPooling is applied after each R2CL block to progressively reduce the spatial dimensions, enabling the extraction of deeper, high-level features.

### Bottleneck Layer

- The bottleneck is the deepest part of the network, responsible for processing the most abstract and compact feature representations of the input data. The bottleneck also utilizes R2CL blocks to further refine the extracted features.

### Upsampling Path (Decoder)

- In the upsampling path, the model uses transpose convolutions to increase the spatial resolution of feature maps, bringing them back to the original input size.
- The Attention Gates (AGs) are applied here to selectively highlight relevant features while suppressing irrelevant information. This allows the model to focus on tumor regions more effectively, improving localization accuracy during segmentation.
- The upsampled features are concatenated with corresponding feature maps from the encoder (skip connections) to maintain fine-grained details.

### Output Layer

- The final output layer uses a 1x1 convolution to reduce the number of channels to one and applies a sigmoid activation function to generate a binary segmentation mask. This mask highlights the tumor regions within the brain MRI images.

### Batch Normalization (BN)

- Throughout the architecture, Batch Normalization (BN) layers are employed to stabilize the learning process by normalizing the input to each layer, leading to faster convergence and better generalization.

### Key Advantages of the Architecture

- **R2CL Blocks**: Recurrent residual connections enable efficient feature reuse, capturing complex spatial patterns within the MRI images and improving learning stability.
- **Attention Gates (AGs)**: Focus on the most critical areas of the image, allowing for enhanced segmentation accuracy, particularly in regions where tumor boundaries may be difficult to detect.
- **Efficient Design**: Despite its advanced architecture, Mod-R2AU-Net reduces the parameter count and computational complexity, making it suitable for deployment in resource-constrained environments.

The Mod-R2AU-Net architecture, as shown, is designed to achieve state-of-the-art performance for brain tumor segmentation while maintaining computational efficiency, ensuring its suitability for both research and clinical applications.

### Graphical Abstract

![Graphical Abstract](https://github.com/user-attachments/assets/2459cf54-f241-4a98-92bc-cac24d3b8c0a)

I used several metrics to track the progress of training for the proposed model, including Accuracy, Loss, Dice Coefficient, and Intersection over Union (IoU). By monitoring these metrics throughout the training process, we gain a comprehensive understanding of the model’s performance. This approach allows us to make necessary adjustments and improvements to ensure that the Mod-R2AU-Net model achieves optimal accuracy and reliability in brain tumor segmentation.
![Results](https://github.com/user-attachments/assets/3332b476-58c2-42ce-a607-1fcb3847cff6)
Training and Validation Metrics (Accuracy, Loss, Dice Coefficient, IoU) on BRATS 2020 Dataset.
