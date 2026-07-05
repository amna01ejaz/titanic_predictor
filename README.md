# Titanic Survival Prediction

This project implements a complete machine learning workflow to predict passenger survival outcomes from the Titanic disaster based on demographic and ticketing features such as age, sex, class, and family size.

## Dataset
The project utilizes the training data from the Titanic: Machine Learning from Disaster competition on Kaggle.

## Project Structure
* app.py: The primary Python script handling missing value imputation, categorical feature encoding, feature engineering, and logistic regression modeling.
* train.csv: The baseline training dataset containing passenger attributes and survival target labels.

## Prerequisites and Installation
Ensure your environment contains the required data processing and machine learning libraries:

```bash
pip install pandas scikit-learn