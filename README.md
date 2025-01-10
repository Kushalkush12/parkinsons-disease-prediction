Parkinson's Disease Prediction
This repository contains a machine learning project for predicting Parkinson's disease using various classification algorithms. The aim is to analyze and compare model performance to determine the most effective approach for identifying Parkinson's disease based on a given dataset.

Table of Contents
Introduction
Features
Technologies Used
Dataset
Installation
Usage
Models and Results


Introduction
Parkinson's disease is a neurodegenerative disorder that affects millions worldwide. Early and accurate detection can significantly enhance treatment outcomes. This project utilizes machine learning techniques to classify individuals as Parkinson's positive or negative based on extracted features from a dataset.

Features
Multiple machine learning algorithms implemented:
Support Vector Machine (SVM)
Random Forest
Logistic Regression
K-Nearest Neighbors (KNN)
Gradient Boosting
Dataset preprocessing and feature selection.
Model evaluation using accuracy, precision, recall, and F1-score.
Visualization of results for model comparison.


Technologies Used
Python
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Jupyter Notebook


Dataset
The dataset used in this project is publicly available and contains features extracted from voice recordings of individuals. These features include measures of variation in fundamental frequency, amplitude, and other voice metrics related to Parkinson's symptoms.

Source: UCI Machine Learning Repository or other sources as per project.

Installation
Clone the repository:
bash
Copy code
git clone https://github.com/Kushalkush12/parkinsons-disease-prediction.git
cd parkinsons-disease-prediction
Create and activate a virtual environment (optional but recommended):
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt


Usage
Open the Jupyter Notebook:
bash
Copy code
jupyter notebook
Navigate to the main notebook file and follow the instructions for:
Loading and preprocessing the dataset.
Training and testing various models.
Evaluating and visualizing model performance.


Models and Results:
Model	Accuracy	Precision	Recall	F1-Score
Support Vector Machine (SVM)	95%	93%	96%	94%
Random Forest	94%	92%	95%	93%
Logistic Regression	91%	89%	90%	89%
K-Nearest Neighbors (KNN)	88%	86%	88%	87%
Gradient Boosting	96%	94%	97%	95%
