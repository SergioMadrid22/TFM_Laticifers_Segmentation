# Medical Image Segmentation Workflow

## Configuration Management

Training hyperparameters are defined in YAML configuration files. Reference example: `configs/unet_patched.yaml`

## Training Pipeline (Patch-based)

### Data Preparation
- **Patch Selection**: Random sampling from input images
- **Input Processing**: 
  - Base input: Original grayscale images enhanced with CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - **Custom Features**: Additional input channels can be added by passing a dictionary of `{feature_name: directory_path}` pairs to the dataset constructor. This enables integration of features derived from classical image processing methods
- **Quality Control**: Optional vessel proportion threshold for intelligent patch selection (filters patches based on vessel content)

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Loss Functions**: 
  - Single loss: Choose from BCE, Dice, IoU, Tversky, clDice
  - Dual loss: Linear combination of any two available loss functions

## Inference Pipeline (Patch-based)

### Sliding Window Approach
- **Window Size**: Matches training patch dimensions
- **Stride**: Typically set to half of patch size for optimal overlap
- **Reconstruction**: Overlapping pixel predictions are averaged to generate final segmentation

## Loss Function Options

Available loss functions for vessel segmentation:
- **BCE**: Binary Cross-Entropy
- **Dice**: Dice Similarity Coefficient
- **IoU**: Intersection over Union
- **Tversky**: Tversky Loss (generalizes Dice)
- **clDice**: Centerline Dice (topology-aware loss for tubular structures)

## Future Enhancement Techniques

### Data Augmentation & Generation
- **Test Time Augmentation (TTA)**: Apply multiple augmentations during inference and average predictions
- **Synthetic Data Generation**: GAN-based data augmentation for limited datasets (Reference: "Patient‑specific placental vessel segmentation with limited data")

### Post-processing Methods
- **Dense CRF**: Conditional Random Fields for boundary refinement
- **Morphological Operations**: Opening, closing, and other morphological filters
- **Binarization**: Threshold-based conversion to binary masks

### Semi-supervised Learning
- **Pseudolabels**: Use model predictions on unlabeled data as additional training samples

## Implementation Notes

- Ensure YAML configuration files follow the structure defined in `configs/unet_patched.yaml`
- Custom feature integration requires consistent directory structure and naming conventions
- Patch-based approach is particularly effective for high-resolution medical images where memory constraints limit full-image processing