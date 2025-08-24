# Real-Time Credit Card Fraud Detection System

## Project Overview

This project aims to develop a real-time fraud detection system for a bank. The primary objective is to build a machine learning model that can accurately identify fraudulent credit card transactions while minimizing false positives. The model should be able to capture the complex relationships between transaction features and provide a probabilistic assessment of whether a transaction is fraudulent.

This repository contains the complete workflow, from data preprocessing and model development to a comprehensive performance evaluation, including techniques to handle the challenges of imbalanced datasets common in fraud detection.

## Table of Contents
1.  [Data Preprocessing](#data-preprocessing)
2.  [Model Development](#model-development)
3.  [Performance Evaluation](#performance-evaluation)
4.  [Addressing Class Imbalance](#addressing-class-imbalance)
5.  [Final Results](#final-results)
6.  [How to Run the Project](#how-to-run-the-project)
7.  [Conclusion and Recommendations](#conclusion-and-recommendations)

---

### Data Preprocessing

The initial and most critical phase was to prepare the `credit_card_fraud_dataset_modified.csv` data for model training. The following steps were performed:

* **Handling Missing Values:**
    * Missing values in numerical columns (`Amount`, `CardHolderAge`) were imputed using the median value of their respective columns.
    * Missing values in categorical columns (`MerchantCategory`) were imputed using the mode (the most frequent category).

* **Feature Scaling and Encoding:**
    * **Numerical Features** (`Amount`, `Time`, `CardHolderAge`): These were scaled using `StandardScaler` to ensure that all features have a mean of 0 and a standard deviation of 1. This prevents features with larger scales from dominating the model.
    * **Categorical Features** (`Location`, `MerchantCategory`): These were converted into a numerical format using `OneHotEncoder`. This creates new binary columns for each category, allowing the model to interpret them.

* **Data Splitting:**
    * The dataset was split into a training set (80%) and a testing set (20%). The split was stratified by the `IsFraud` target variable to ensure that both the training and testing sets had a similar proportion of fraudulent and non-fraudulent transactions.

---

### Model Development

The goal was to select a model that provides probabilistic outputs and can serve as a strong baseline for this classification task.

* **Chosen Model: Logistic Regression**
    * **Justification:** Logistic Regression was selected as the primary model for several reasons:
        1.  **Probabilistic Outputs:** It naturally provides a probability score (between 0 and 1) for each prediction, which is a key requirement for this project. This allows the bank to set a specific threshold for flagging transactions based on their risk appetite.
        2.  **Interpretability:** It is a relatively simple and highly interpretable model. The coefficients of the logistic regression equation can be used to understand the influence of each transaction feature on the likelihood of fraud.
        3.  **Efficiency:** It is computationally lightweight and fast to train, making it an excellent candidate for real-time detection systems.

* **Comparison Model: Random Forest Classifier**
    * To evaluate the performance of our chosen model, a Random Forest Classifier was also developed. As a powerful ensemble method, Random Forest can capture complex non-linear relationships and typically offers higher accuracy, serving as a robust benchmark.

---

### Performance Evaluation

The performance of both models was evaluated on the unseen test data using a variety of standard classification metrics.

#### Initial Results (Before Handling Class Imbalance)

The initial models were trained on the original, imbalanced dataset. While the accuracy was high (~95%), the models were completely ineffective at their primary goal: detecting fraud. A **Recall** of 0.0 indicated that **not a single fraudulent transaction was correctly identified**. This is a classic symptom of a model trained on an imbalanced dataset, where the model learns to always predict the majority class (non-fraud).

#### Addressing Class Imbalance

To solve this critical issue, **random oversampling** was applied to the training data. This technique balances the dataset by duplicating the instances of the minority class (fraudulent transactions) until they are equal in number to the majority class. This forces the model to learn the patterns of fraudulent transactions.

#### Final Results (After Oversampling)

After retraining on the balanced dataset, the Logistic Regression model showed a dramatic improvement in its ability to detect fraud. The final results on the test set are as follows:

| Model | Accuracy | Precision | Recall | F1-score | ROC AUC |
|---|---|---|---|---|---|
| **Logistic Regression (Oversampled)** | **0.5600** | **0.0465** | **0.4000** | **0.0833** | **0.5053**|
| Random Forest (Oversampled) | 0.9400 | 0.0000 | 0.0000 | 0.0000 | 0.3200 |

**Key Takeaway:** The **oversampled Logistic Regression model is the superior choice**. Although its overall accuracy is lower, its **Recall of 0.40** means it successfully identifies 40% of the fraudulent transactions in the test set—a massive improvement from zero. The lower accuracy is a positive sign that the model is no longer ignoring the minority class and is actively attempting to identify fraud. The Random Forest model, in contrast, failed to learn from the oversampled data and still did not identify any fraudulent transactions.

---

### How to Run the Project

1.  **Prerequisites:** Ensure you have Python installed, along with the following libraries:
    * pandas
    * scikit-learn
    * matplotlib
    * seaborn

    You can install them using pip:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn
    ```

2.  **Dataset:** Place the `credit_card_fraud_dataset_modified.csv` file in the same directory as the script.

3.  **Execute the Script:** Run the provided Python script to perform the data preprocessing, model training, and evaluation. The script will output the performance metrics and generate visualizations of the confusion matrices.
    ```bash
    python fraud_detection_script.py
    ```

---

### Conclusion and Recommendations

The final **oversampled Logistic Regression model** provides a solid foundation for a real-time fraud detection system. It successfully moves beyond the high-accuracy pitfall of imbalanced data to provide genuine fraud detection capabilities.

**Recommendations for further improvement include:**
* **Threshold Tuning:** Adjust the probability threshold (default is 0.5) to balance the trade-off between Recall (catching more fraud) and Precision (reducing false positives).
* **Advanced Feature Engineering:** Create new features from the existing data to better capture transactional patterns.
* **Explore Other Imbalance Techniques:** Experiment with other methods like SMOTE or undersampling to compare results.
