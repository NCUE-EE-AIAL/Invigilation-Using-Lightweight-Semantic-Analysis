# Invigilation Using Lightweight Semantic Analysis

This repository contains the implementation of a vision-based invigilation system designed to enhance examination security through **lightweight semantic analysis**. The project utilizes **Swin Transformer Tiny** for semantic segmentation and **ShuffleNet V2** for facial recognition, providing an efficient, scalable solution for detecting examination cheating.

## 📜 Overview

### Key Features
- **Lightweight Deployment**: Designed to operate efficiently on edge computing devices, making it suitable for use in constrained environments such as examination halls.
- **High Accuracy Models**:
  - **Semantic Segmentation with Swin-T**: Achieves **92.74% accuracy** for precise object boundary detection.
  - **Facial Recognition with ShuffleNet V2**: Attains **98.58% accuracy** in verifying student identities.
- **Real-Time Monitoring**: Capable of handling complex scenarios like multiple students or various types of unauthorized items, both in traditional and computer-based exam settings.

### Abstract
The project presents a system that integrates advanced **deep learning models** to effectively monitor students during examinations and detect cheating behaviors. The lightweight architecture is optimized for **edge deployment**, combining powerful deep learning tools to provide real-time detection of unauthorized activities while minimizing computational overhead.

## 📂 Datasets
- **Classification Dataset**: A proprietary dataset comprising images from **13 individuals**, each contributing 30 photographs. We used **MTCNN** for facial feature extraction, significantly improving model convergence.
- **Segmentation Dataset**: We utilized the **CHIP-19 dataset** containing comprehensive human body information for training our Swin Transformer model.

## 🔍 Methodology
- **ShuffleNet V2**:
  - A lightweight convolutional neural network used for **identity classification**.
  - Features an architecture optimized for resource-constrained devices, maintaining a memory footprint of only **36.87 MB**.
- **Swin Transformer Tiny (Swin-T)**:
  - A vision transformer designed for **semantic segmentation**, utilizing **shifted window-based self-attention** to accurately detect object boundaries.
  - Suitable for both **fixed** and **dynamic camera setups**, making it versatile for examination environments.

### Decision-Level Fusion
To enhance system performance, **decision-level fusion** was implemented:
- **Segmentation Model**: Responsible for delineating distinct objects, such as arms and unauthorized items.
- **Classification Model**: Focuses on identifying student identities, especially in scenarios where individual actions need to be correlated with identity.

This separation improves overall system efficiency and minimizes model complexity.

## ⚙️ System Setup
The system operates in two examination environments:
1. **Computer-Based Exams**:
   - Requires only **basic camera** and **6GB RAM**.
   - Monitors identity through periodic face verification and detects **contraband** such as earphones or smartphones.
2. **Traditional Exams**:
   - Utilizes both **fixed and dynamic cameras** to capture contraband and student movements.
   - A dynamic camera, often mounted on a **robot**, navigates to monitor the entire room.

## 🧩 Flow Diagram for Action Detection
Below is the flow diagram representing the algorithm used for action detection in traditional examination settings. It shows how **face identification**, **object detection**, **segmentation**, and **pixel analysis** are utilized to detect irregular actions, such as mismatched student identities, the presence of contraband, and suspicious movements.

```mermaid
graph TD
    A[Real-time Image] --> B(Face Identification)
    A --> C(Object Detection - Extends)
    A --> D(Segmentation)
    B --> E{Student ID different with face}
    E -->|Yes| F[Warn]
    E -->|No| G{Contraband Exist}
    G -->|Yes| F
    G -->|No| H[Pass & Return]
    D --> I[Pixel Algorithm]
    I --> J{People label attach Boundary}
    J -->|No| K{Eyes boundary From 2 to 1 (Turn around)}
    K -->|Yes| F
    K -->|No| H
    J -->|Yes| F
```

The flow diagram provides a comprehensive overview of the steps taken during real-time monitoring, highlighting key decision points to detect and respond to potential cheating behaviors.

## 📊 Results
- **Semantic Segmentation**: Achieved high accuracy in delineating full human bodies and detecting hands in real-time.
- **Facial Classification**: ShuffleNet V2 effectively identified students in examination scenarios, even with limited training data.

## 💡 Applications
This project has significant implications for:
- **Educational Environments**: Increasing exam security by minimizing manual monitoring needs and reducing cheating incidents.
- **Edge Computing**: Demonstrates the potential for using deep learning models in constrained environments where **computational efficiency** is critical.

## 🚀 Getting Started
To get started with the code, clone this repository:

```sh
git clone https://github.com/yourusername/invigilation-lightweight-semantic.git
```

Install the dependencies:

```sh
pip install -r requirements.txt
```

## 📝 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments
This project was supported by the **National Changhua University of Education** and was developed under the guidance of **Prof. Wen-Ran Yang**.
