# DeepSurv-Relapse-Prediction

This is the official code repository for the manuscript:
**"Predicting Time-to-Relapse Using Deep Survival Analysis: Unveiling Non-Linear Risk Dynamics and the Impulsivity Threshold in a Single-Center Cohort"** (Submitted to JMIR).

## Overview
This repository contains the complete machine learning pipeline used in our study, including:
* Data preprocessing and standardization.
* Model training for **DeepSurv**, **Cox Proportional Hazards (CoxPH)**, and **Random Survival Forest (RSF)**.
* Survival performance evaluation (Time-dependent C-index, Integrated Brier Score, ICI, E50).
* Explainable AI (XAI) analysis using **SHAP** to extract non-linear risk thresholds (e.g., BIS-11) and visualize the "protective buffer effect" of Family Support (FSI).

## Requirements
To run this pipeline, please ensure you have the following packages installed:
`pip install pycox scikit-survival lifelines shap optuna torch pandas numpy matplotlib`

## Data Privacy
Due to strict institutional and regulatory restrictions regarding the sensitive nature of public security and drug rehabilitation records in China, the original clinical dataset (N=3759) cannot be made public. 
However, we provide the full execution script (`main_analysis.py`) which can be run transparently on similar structural data.
