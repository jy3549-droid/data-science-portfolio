# data-science-portfolio

Machine Learning projects for QMSS program.

Iris Classification: Multilayer Perceptron (MLP) with Keras

Project Overview

This repository contains the implementation of a deep learning project focused on building, training, and evaluating Multilayer Perceptron (MLP) models using the Keras/TensorFlow framework. The goal is to classify the species of the classic Iris flower dataset.

The core objective of this project is to explore the trade-offs between network depth, architectural complexity, and generalization ability when designing neural networks.

Technical Stack

Core Language: Python

Deep Learning Framework: Keras (TensorFlow backend)

Data Handling: Pandas, NumPy

Modeling: Scikit-learn (Preprocessing)

Methodology and Model Architecture

1. Data Preprocessing

The Iris dataset was loaded and split into training and testing sets.

Numerical features were preprocessed (e.g., scaling).

Target labels (flower species) were One-Hot Encoded to fit the multi-class classification requirement.

2. Comparative Model Design

Three distinct MLP architectures were designed and compared to assess the impact of network capacity:

Model Name	Architecture (Hidden Units per Layer)	Key Focus
Model 1 (Baseline)	Moderate size (e.g., two 16-unit layers)	Serves as the standard reference for performance.
Model 2 (Smaller)	Shallow/Fewer units (e.g., 8-8 units)	Exploration of underfitting risks due to limited model capacity.
Model 3 (Deeper)	Deeper and larger (e.g., 32-16-8 units)	Assessment of increased representational power and potential for overfitting.
All models were trained using the Adam optimizer and Categorical Cross-Entropy loss for 30 epochs.

Key Results and Findings

Model	Test Set Accuracy	Analysis
Model 1	0.9667	Strong performance for a moderately sized network.
Model 2	0.9333	Reduced capacity led to a slight drop in accuracy, demonstrating slight underfitting.
Model 3	1.00	Achieved perfect accuracy. This highlights the network's high representational power, but in a small dataset context, the risk of overfitting must be acknowledged.
Conclusion: The experiment clearly illustrates how increasing network depth and unit count enhances model fitting capability. It demonstrates an ability to implement and interpret model performance metrics relative to architectural design.

Skills Demonstrated

This project effectively validates the following career-relevant technical skills:

Deep Learning Implementation: Hands-on proficiency with the Keras/TensorFlow framework.

Neural Network Design: Ability to define, compile, and train varying MLP architectures.

Model Evaluation: Understanding of training dynamics, model complexity, and the critical skill of assessing overfitting risk in deep learning models.







Clickbait Headline Classification: NLP and Logistic Regression

Project Overview

This project focuses on building a binary text classification model to distinguish between clickbait and non-clickbait news headlines. The goal is to apply fundamental Natural Language Processing (NLP) techniques and machine learning to a real-world linguistic classification problem.

The solution utilizes Logistic Regression after careful text vectorization and feature engineering, demonstrating proficiency in both model development and interpreting feature importance.

Technical Stack

Core Language: Python

NLP & ML: Scikit-learn (Logistic Regression, GridSearchCV, Feature Extraction)

Data Handling: Pandas, NumPy

Visualization (Optional): Matplotlib/Seaborn (for coefficient visualization)

Methodology and Feature Engineering

The strength of this project lies in its feature engineering process, which directly influences the model’s performance and interpretability:

Data Preprocessing: Standard text cleaning (tokenization, stemming/lemmatization, stop-word removal) was applied.

Feature Vectorization: The model was trained using a comparative approach to text representation:

Bag-of-Words (BoW): To establish a baseline frequency model.

TF-IDF (Term Frequency-Inverse Document Frequency): To weigh word importance, balancing signal and noise.

N-Grams: Applied to capture short sequences of words (e.g., bigrams, trigrams) to identify common clickbait patterns.


Model Training & Tuning:

A Logistic Regression model was selected for its performance and high interpretability.

GridSearchCV was utilized for systematic hyperparameter tuning (e.g., the regularization strength parameter, C) to optimize the model’s performance against overfitting.

Key Results and Interpretability

Performance: The optimized model achieved a competitive performance metric, such as an F1-Score of approximately 0.93 (or the exact F1-score from your final model), demonstrating high precision and recall in classification.

Interpretability: A critical component of this project was analyzing the model coefficients. This analysis revealed which specific words or phrases were the strongest positive (clickbait indicators) and negative (non-clickbait indicators) predictors.


Skills Demonstrated

This project effectively validates the following career-relevant technical skills:

Natural Language Processing (NLP): Practical application of core text vectorization techniques (BoW, TF-IDF, N-Grams).

Model Tuning: Proficient use of GridSearchCV and cross-validation for hyperparameter optimization.

Feature Engineering: Strategic design and comparison of text features to maximize model performance.

Model Interpretability: Ability to analyze and explain model coefficients to derive actionable insights, which is crucial in applied data science.
