# Heart Murmur recognition with Deep Learning

## Overview
This repository contains the code and experiments for a **heart murmur detection system** using deep learning on phonocardiogram (PCG) recordings.  
The goal is to classify each heart sound recording into one of three categories:
- **Present** – murmur is detected
- **Absent** – no murmur detected
- **Unknown** – uncertain or poor-quality recording

The work builds on the **CirCor DigiScope Phonocardiogram Dataset** and explores multiple neural network architectures, feature extraction techniques, and evaluation settings.

## Dataset
We used the **CirCor DigiScope Phonocardiogram Dataset (PhysioNet v1.0.3)**, which contains heart sound recordings from multiple auscultation locations, collected in clinical settings.  
For most experiments, only the **publicly available training portion** of the dataset was used.  
Multi-location recordings were segmented, preprocessed, and converted into spectrograms before training.

## Models
We evaluated several architectures:
- **CNN-based classifier**
- **ResNet-based models** (ResNet18 and ResNet34, adapted for 1-channel or 3-channel spectrograms)
- **Attention-based CNN classifier**

## Feature Extraction
- **Mel spectrograms** were extracted from PCG recordings.
- Both **single-window** and **multi-window** spectrogram approaches were tested.
- Normalization and augmentation were applied to improve generalization.


