# Fraudulent-Transaction-Detection-in-credit-card-activity

## Overview

This project aims to detect fraudulent transactions in credit card activity using machine learning models. The dataset contains anonymized features, V1-V28, along with `Time`, `Amount`, and `Class`, where:
- `Class 0`: Not fraudulent
- `Class 1`: Fraudulent

## Project Structure

- **Data Loading and Preprocessing**:
  - The dataset is loaded from Google Drive.
  - Initial data exploration includes distribution plots and histograms of the `Amount` and `Time` features.
  - The dataset is split into training and testing sets, and features are standardized.

- **Exploratory Data Analysis (EDA)**:
  - Visualization of feature distributions and relationships between `Time`, `Amount`, and `Class`.
  - Examination of the imbalance between classes and its handling using resampling techniques.

- **Model Training**:
  - Several machine learning models are trained including Logistic Regression, Decision Trees, and Neural Networks.
  - The performance of each model is evaluated using metrics such as accuracy, precision, recall, and F1-score.

- **Model Evaluation**:
  - Confusion matrices and precision-recall curves are used to assess model performance.
  - Visualization of training and validation accuracy and loss for neural network models to detect overfitting.

## Key Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- keras

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/FraudulentTransactionDetection.git
   cd FraudulentTransactionDetection
   ```

2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   Open the `FradulentProject.ipynb` notebook in Jupyter and execute the cells sequentially.

## Dataset

The dataset used for this project is the Credit Card Fraud Detection dataset from Kaggle, which is not included in this repository. You can download it from [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the appropriate directory.

## Results

The project involved training multiple models to detect fraudulent transactions. The performance of each model is summarized below:

1. **Logistic Regression**:
   - **Accuracy**: 99.28%
   - **Precision**: 90.91%
   - **Recall**: 69.92%
   - **F1-Score**: 78.96%

2. **Decision Tree**:
   - **Accuracy**: 99.92%
   - **Precision**: 97.67%
   - **Recall**: 87.69%
   - **F1-Score**: 92.38%

3. **Random Forest**:
   - **Accuracy**: 99.95%
   - **Precision**: 98.72%
   - **Recall**: 90.77%
   - **F1-Score**: 94.56%

4. **Support Vector Machine (SVM)**:
   - **Accuracy**: 99.93%
   - **Precision**: 97.91%
   - **Recall**: 89.23%
   - **F1-Score**: 93.37%

5. **Neural Network**:
   - **Accuracy**: 99.44%
   - **Precision**: 92.31%
   - **Recall**: 80.77%
   - **F1-Score**: 86.11%

The neural network model achieved a high accuracy of 99.44%, with balanced precision and recall values, indicating good generalization. The Random Forest model performed slightly better in terms of accuracy and F1-Score but may require more computational resources. The Support Vector Machine (SVM) model also showed high performance with a good balance between precision and recall.

Overall, the Random Forest and Decision Tree models provided the best performance in terms of accuracy and precision, while the neural network model offered a good balance and generalization capability.


## Contributions

Contributions to this project are welcome. Please create a pull request or open an issue for any suggestions or improvements.
