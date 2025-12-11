# Model B: Multimodal Emotion Recognition with Cross-Attention Fusion

Model B is a transformer-based multimodal emotion recognition system that combines facial imagery and textual statements to predict emotional states. The architecture uses ResNet-18 for visual features and a transformer encoder for textual features, fused through cross-attention mechanisms inspired by Visual Question Answering (VQA) architectures.

## Architecture Overview

- **Image Branch**: Pretrained ResNet-18 → 512-dimensional image features
- **Text Branch**: Transformer encoder (3 layers, 4 heads) → 512-dimensional text embedding
- **Fusion**: Cross-attention (text attends to image) + Concatenation → 1024-dimensional vector
- **Classifier**: Linear layer (1024 → 7 emotion classes)
- **Output**: 7-class emotion prediction (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)

## Files Included

- `ModelB.ipynb` - Main notebook containing all code for training, evaluation, and demonstrations
- `test_results.csv` - Test set evaluation results (accuracy, precision, recall, F1 scores)
- `requirements.txt`

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- transformers
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- Kaggle API (for dataset download)

## Quick Start

### 1. Setup

1. Open `ModelB.ipynb` in Jupyter Notebook or Google Colab
2. Configure the "SETUP & CONFIGURATION" section:
   - Set `RUN_IN_KAGGLE = True` if using Kaggle, or `False` for local/Colab
   - Adjust hyperparameters as needed (learning rate, dropout, epochs, etc.)

### 2. Data Preparation

The notebook automatically downloads datasets:
- **RAF-DB** (Balanced, grayscale): Facial emotion images
- **Emotions Dataset**: Text samples labeled with emotions

Datasets are pseudo-paired by matching emotion labels.

### 3. Training

**Option A: Train a New Model**
1. Run all cells up to "Training Loop (Option A: Train New Model)"
2. Execute the training loop cell
3. Model checkpoints are saved in `checkpoints/` directory

**Option B: Load Existing Checkpoint**
1. Run all cells up to "Load Saved Model (Option B: Use Existing Checkpoint)"
2. Set `EXPERIMENT_ID` to match your checkpoint (e.g., 14 for `modelB_exp014.pth`)
3. Execute the load checkpoint cell

### 4. Evaluation

Run the "Test Evaluation" section to:
- Evaluate on test set
- Generate confusion matrix
- Calculate per-class metrics (precision, recall, F1)
- Save results to `test_results.csv`

### 5. Demonstrations

The notebook includes demonstration sections:
- **Test Set Examples**: Random samples from test set with predictions
- **Custom Tests**: Test on your own images and text inputs

## Key Hyperparameters (Best Configuration)

- **Learning Rate**: 0.0001
- **Batch Size**: 64
- **Epochs**: 100
- **Dropout**: 0.2
- **Transformer Layers**: 3
- **Attention Heads**: 4
- **Cross-Attention**: Enabled
- **Optimizer**: AdamW
- **Loss Function**: Cross-Entropy

## Performance

Best configuration (Experiment 014):
- **Test Accuracy**: 83.19%
- **Macro F1 Score**: 0.8320
- **Macro Precision**: 0.8328
- **Macro Recall**: 0.8319

See `test_results.csv` for detailed per-class metrics.

## Notebook Structure

1. **SETUP & CONFIGURATION** - All hyperparameters and settings
2. **DATA PREPARATION** - Dataset loading, preprocessing, transforms
3. **MODEL ARCHITECTURE** - Transformer encoder and multimodal classifier definitions
4. **TRAINING & EVALUATION** - Training loop, checkpoint loading, test evaluation
5. **DEMONSTRATIONS** - Visual examples and custom testing

## Notes

- The notebook is organized with clear sections and instructions
- Model checkpoints are saved automatically during training
- Training takes approximately 1 hour on NVIDIA RTX 5060 Ti GPU for 100 epochs
- ResNet-18 backbone is frozen (pretrained weights only) for faster training

## Citation

If you use this model, please cite the associated paper (see main repository README for citation details).

