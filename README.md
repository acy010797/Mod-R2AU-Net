# Mod-R2AU-Net for Brain Tumor Segmentation
Mod-R2AU-Net is an optimized and enhanced version of the U-Net architecture, specifically designed for advanced medical image segmentation, with a focus on brain tumor segmentation in MRI images. The model integrates Recurrent Residual Convolutional Layers (R2CL), Attention Gates (AGs), and Batch Normalization (BN) to significantly improve feature extraction, processing speed, and segmentation accuracy.

The R2CL blocks are employed in both the encoder and decoder paths, utilizing residual connections and recurrent learning to refine feature representation over multiple iterations. This allows the model to capture complex patterns in MRI images more effectively. Attention Gates (AGs) are applied to further enhance the segmentation process by focusing on the most relevant regions within the image, improving the model’s ability to distinguish between healthy and tumor tissues.

Batch Normalization (BN) is incorporated to stabilize and accelerate the training process, ensuring that each layer’s input is appropriately scaled and distributed. This results in faster convergence and improved generalization, making the model more robust in various segmentation tasks.

Despite its advanced architecture, Mod-R2AU-Net remains computationally efficient, reducing the number of parameters and computational demands while delivering high accuracy. With 212 GFlops and 21.3M parameters, it is a lightweight model suitable for deployment in real-world clinical settings. These optimizations allow for faster processing times and resource-friendly deployment on modern hardware, such as GPUs, making the model ideal for large-scale MRI datasets. Mod-R2AU-Net’s combination of R2CL, AGs, and BN ensures a strong balance between computational efficiency and precise brain tumor segmentation, making it a powerful tool for medical professionals and researchers.
# Model Architecture
![Graphical Abstract](https://github.com/user-attachments/assets/2459cf54-f241-4a98-92bc-cac24d3b8c0a)
Proposed architecture of Mod-R2AU-Net for brain tumor segmentation.
