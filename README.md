# NeuroCardIQ: Multimodal Brain–Heart Risk Assessment System

## Overview

NeuroCardIQ is an AI-powered mental health risk assessment system that analyzes both brain activity (EEG) and heart activity (ECG) signals to identify potential signs of mental health deterioration. By combining physiological information from the brain and heart, the system provides a more comprehensive and objective assessment compared to traditional single-modality approaches.

The project leverages machine learning and explainable AI techniques to generate transparent predictions, helping users better understand the factors influencing mental health risk assessments.

---

## Problem Statement

Mental health disorders such as stress, anxiety, and depression often go undetected in their early stages because traditional assessment methods rely heavily on questionnaires, self-reporting, and clinical observation.

NeuroCardIQ addresses this challenge by using physiological signals that can provide objective indicators of emotional and psychological states. The goal is to develop an intelligent system capable of detecting mental health risks through the combined analysis of EEG and ECG data.

---

## Key Features

* Brain–Heart Interaction Analysis using EEG and ECG signals
* Multimodal Feature Fusion for improved prediction performance
* Mental Health Risk Classification using Machine Learning
* Explainable AI (SHAP) for transparent model predictions
* Interactive Dashboard for result visualization
* Prediction History Tracking and User Analytics
* Personalized Recommendation System

---

## Dataset

**DREAMER Dataset**

The project uses the DREAMER dataset, which contains synchronized EEG and ECG recordings collected from participants under emotional stimulation conditions. The dataset enables the study of relationships between brain activity, heart activity, and emotional responses.

---

## Methodology

1. Data Collection and Understanding
2. Signal Preprocessing and Cleaning
3. EEG and ECG Feature Extraction
4. Brain–Heart Interaction Feature Generation
5. Feature Fusion and Scaling
6. Machine Learning Model Training
7. Model Evaluation and Comparison
8. Explainable AI Analysis using SHAP
9. Dashboard-Based Result Visualization

---

## Models Evaluated

* Random Forest
* XGBoost

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

---

## Results

| Model         | Accuracy |
| ------------- | -------- |
| Random Forest | 97.09%   |
| XGBoost       | 97.58%   |

XGBoost achieved the best overall performance and was selected as the final prediction model.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Flask
* SQLite
* Matplotlib

---

## My Contributions

* Conducted research on mental health assessment using EEG and ECG signals.
* Worked with the DREAMER dataset, including preprocessing and data preparation.
* Experimented with multiple machine learning models and compared their performance.
* Assisted in model evaluation and result interpretation.
* Collaborated with teammates throughout development using AI-assisted tools for faster experimentation and implementation.
* Contributed to project documentation, testing, and presentation preparation.

---

## Future Enhancements

* Integration with real-time wearable EEG and ECG devices
* Deep learning-based feature extraction
* Cloud deployment for scalability
* Continuous mental health monitoring
* Clinical validation using larger datasets
