# Histopathology Out-of-Distribution (OOD) Classification

### MVA Project DLMI 2025 — Group 37  
**Authors:** Lucas Murray-Hidalgo, Antoine Sicard  

## 🧠 Project Overview

This project addresses the challenge of classifying histological images as cancerous or non-cancerous across different medical centers. Domain shift caused by heterogeneous data sources often hinders model generalization. To tackle this, we explore domain adaptation techniques with a strong emphasis on **adversarial learning** and **self-supervised pretraining** using **DINOv2 (ViT-S/8)**.

---

## 🔍 Motivation

Deep learning models can achieve remarkable results in cancer detection but struggle to generalize when trained on data from a single source. The project aims to:
- Learn **domain-invariant representations**.
- Mitigate **center-specific artifacts**.
- Improve **generalization to unseen data** through self-supervised learning and domain adversarial strategies.

---

## 🧱 Methodology

### 1. **Self-Supervised Pretraining**
- **Backbone:** DINOv2 (ViT-S/8), pretrained on ImageNet.
- **Augmentation:** Standard DINO-style augmentations.
- **Domain Discriminator:** MLP + Gradient Reversal Layer to discourage center-specific features.
- **Loss Function:**
  ```
  L_total = L_DINO + α(t) * L_adv
  ```
  where `α(t)` increases over time using a sigmoid schedule.

### 2. **Supervised Learning**
- **Classifier Head:** MLP on top of the DINO backbone.
- **Two strategies tested:**
  - Frozen backbone
  - Partial unfreezing of deeper layers

---

## 🧪 Experiments

Three pretraining setups were tested:
1. **Strong Adversarial**
2. **Weak Adversarial**
3. **No Adversarial (Baseline)**

We experimented with varying classifier architectures (1–4 layers, 128–256 dimensions) and different backbone freezing strategies.

**Key Takeaways:**
- Adversarial pretraining significantly improved generalization (↑ accuracy from 0.78 to >0.90).
- No performance gain from deeper classifier heads.
- Partial unfreezing decreased performance—suggesting overfitting on small labeled datasets.

---

## 📊 Results

| Strategy              | Validation Accuracy | Test Accuracy |
|-----------------------|---------------------|----------------|
| Strong Adversarial    | 0.90                | **0.904**      |
| Weak Adversarial      | 0.86                | **0.909**      |
| No Adversarial        | 0.93                | 0.78           |

Note: Test results from Kaggle submissions. Due to limited submission quota, not all configurations were tested.

---

## 🛠️ Architecture Summary

- **Feature Extractor:** DINOv2 ViT-S
- **Classification Head:** MLP (binary classifier)
- **Domain Discriminator:** MLP (5-class center classifier)
- **Gradient Reversal Layer** to enable adversarial training

---

## 📂 Repository Structure

```
src/
├── dino_mod/                    # Core training modules for DINO
│   ├── main_dino_disc.py       # DINO with adversarial discriminator
│   ├── main_dino.py            # Standard DINO training
│   ├── utils.py                # Helper functions
│   └── vision_transformer.py   # ViT architecture definition
│
├── figures/                    # Output figures and plots
├── weights/                    # Model checkpoints and saved weights
│
├── __init__.py
├── eval.py                     # Evaluation utilities
├── models.py                   # Model architecture components
├── pre-train-dino.ipynb        # Notebook for DINO pretraining
├── train_dann.py               # DANN baseline implementation
├── train_dino.py               # Main DINO training script for supervised learning
├── train_mix.py                # DANN hybrid training strategy
├── train-dino.ipynb            # Kaggle training notebook version
└── utils_training.py           # Training utilities and loss functions
```

---

## 📚 References

- Ganin & Lempitsky (2015). *Unsupervised Domain Adaptation by Backpropagation*.
- Oquab et al. (2024). *DINOv2: Learning Robust Visual Features Without Supervision*.
- Esteva et al. (2017). *Dermatologist-level classification of skin cancer*.

---

## 📌 License

This project is licensed under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).
