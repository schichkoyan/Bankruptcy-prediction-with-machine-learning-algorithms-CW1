# Bankruptcy Prediction with Machine Learning

## Overview

This project applies machine learning techniques to predict corporate bankruptcy. It utilizes a dataset containing 96 financial indicators, processed and analyzed using models like Logistic Regression, Support Vector Machines (SVM), and Random Forest Classifiers. The primary aim is to build a reliable predictive model that helps identify potential bankruptcies with high accuracy.

The Random Forest model emerged as the best-performing classifier, with a predictive accuracy of 96.2%. This project provides valuable insights for investors, regulatory bodies, and companies in developing early warning systems for financial risk management.

## Key Features

- **Dataset**: Kaggle-sourced dataset containing 6,819 entries with 96 financial features.
- **Machine Learning Models**:
  - Logistic Regression
  - Support Vector Machines (SVM)
  - Random Forest Classifier (Best performing model with 96.2% accuracy)
- **Techniques**:
  - Data preprocessing: handling missing values, duplicates, and normalizing features.
  - Addressing data imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
  - k-fold Cross-Validation (k=5) for model evaluation.

## Project Structure

- `bankruptcy_perdiction.ipynb`: Jupyter notebook containing the implementation of the machine learning models and all preprocessing steps.
- `REPORT.pdf`: Detailed report on the methodology, results, and evaluations of the predictive models used in the project.

## Prerequisites

To run this project, you will need the following libraries installed:

- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

You can install these dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository and open the Jupyter notebook file `bankruptcy_perdiction.ipynb`.
2. Run the notebook step by step to preprocess the data, train the models, and evaluate their performance.
3. Explore the `REPORT.pdf` file for a detailed analysis of the results.

## Results

- Logistic Regression Accuracy: 90%
- SVM Accuracy: 94.8%
- Random Forest Classifier Accuracy: 96.2% (best performing model)

## Future Work

Future improvements could involve:

- Integrating macroeconomic indicators into the model for a more comprehensive risk assessment.
- Exploring deep learning techniques for more complex, non-linear patterns.
- Using stratified k-fold cross-validation for more precise model validation.

## Authors

Sarkis Chichkoyan
