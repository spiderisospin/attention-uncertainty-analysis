# Uncertainty vs Attention: Analyzing Model Failure in Computer Vision Models

## Overview

This project evaluates whether **prediction uncertainty** or **attention-based explanations (Grad-CAM)** provide a more reliable signal of model error in deep learning systems.

The task is framed as a **multi-label regression problem** on galaxy images, where models predict probabilistic morphological features.

## Key Results

* **Uncertainty vs Error**
  Prediction entropy shows a moderate positive relationship with model error
  *(Spearman = 0.49)*

* **Error Detection**
  Uncertainty identifies high-error predictions with meaningful performance
  *(ROC-AUC = 0.68 on noisy image data)*

* **Attention vs Reliability**
  Grad-CAM–based signals perform near random for error detection
  *(ROC-AUC = 0.51)*

## Main Takeaway

Prediction uncertainty is a **more reliable and actionable signal of model failure** than attention-based explanations in this setting. In practice, this enables production systems to flag or filter uncertain predictions, improving reliability and reducing the risk of incorrect automated decisions.

## Why This Is Not Trivial

It is tempting to state that difficult or noisy images naturally lead to higher error. However, modern neural networks are often **poorly calibrated**:

* They can be **confident but wrong**
* They can be **uncertain but correct**

This means that **uncertainty does not automatically reflect prediction error**

This project shows that, despite this limitation, **uncertainty still contains enough information to meaningfully identify unreliable predictions**, which is a non-trivial and practically useful result.

## Uncertainty as a Reject Option

To evaluate whether uncertainty can improve system reliability, predictions were ranked by entropy and the most uncertain samples were progressively removed.

* Removing high-uncertainty samples **reduces RMSE on the retained predictions**
* The improvement is stronger than random removal, indicating that uncertainty carries useful information about prediction quality

This demonstrates that uncertainty can be used as a **reject-option mechanism**:

* Flag high-risk predictions for review
* Filter low-confidence outputs before deployment
* Improve reliability of accepted predictions

## Methodology

* **Data**: Kaggle Galaxy Zoo images dataset (~60k samples)
* **Models**:

  * ResNet-18
  * EfficientNet-b0
* **Baseline**: Mean prediction per target variable
* **Training**:

  * Loss: Mean Squared Error (MSE)
  * Optimizer: AdamW + cosine annealing
* **Evaluation**:

  * RMSE for prediction error
  * Spearman correlation (uncertainty vs error)
  * ROC-AUC for high-error detection

## Interpretability vs Reliability

* **Grad-CAM**

  * Provides visual explanations
  * Useful for qualitative inspection
  * Not predictive of model error

* **Prediction Entropy**

  * Linked to prediction error
  * Enables identification of unreliable predictions
  * Supports decision-making via filtering


## Repository Structure

```
main_pipeline.ipynb   # End-to-end pipeline and analysis
src/                 # Data loading, training, evaluation utilities, etc.
```

## Summary

This project demonstrates that:

* Not all interpretability methods are useful for reliability
* Model uncertainty, while imperfect, provides a **practical signal for identifying high-risk predictions**
* Filtering high-uncertainty predictions can **improve system-level reliability**
