# wind-turbine-predictive-maintenance-

# Predictive Maintenance for Wind Turbines Using Machine Learning
# Project Description
Renewable energy sources, particularly wind energy, are crucial for reducing the environmental impact of energy production. This project focuses on using machine learning techniques for predictive maintenance of wind turbines to improve operational efficiency and reduce maintenance costs. The objective is to build classification models that can accurately identify potential failures in wind turbine generators based on sensor data.

# Data Description
The data provided consists of 40 predictor variables and 1 target variable, collected through sensors installed across different wind turbine components. The training dataset contains 20,000 observations, while the test dataset contains 5,000 observations.

# Dependencies
Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
# Code Overview
train.csv: Training dataset for model training and tuning.
test.csv: Test dataset for evaluating the performance of the final model.
predictive_maintenance.ipynb: Jupyter Notebook containing the code for data preprocessing, model training, evaluation, and analysis.
# Modeling Approach
Various classification models were implemented, including Logistic Regression, Decision Tree Classifier, AdaBoost Classifier, Gradient Boosting Classifier, Random Forest Classifier, Bagging Classifier, and XGBoost Classifier. The models were trained and tuned using techniques such as oversampling (SMOTE), hyperparameter tuning (RandomizedSearchCV), and feature importance analysis.

# Results
The AdaBoost Classifier tuned using oversampled data achieved the best performance.
F1 score, accuracy, precision, recall, and ROC-AUC score were used to evaluate model performance.
Top three important features identified: V30, V9, and V18.
The model can effectively predict wind turbine failures, helping reduce maintenance costs.
# Future Steps
Experiment with additional feature engineering techniques to enhance model performance.
Explore ensemble methods and advanced machine learning algorithms for better predictive accuracy.
Collect more data and investigate the impact of additional environmental factors on turbine performance.
# Conclusion
This project demonstrates the effectiveness of machine learning in predictive maintenance for wind turbines. By accurately predicting potential failures, the model can help reduce maintenance costs and improve operational efficiency in the renewable energy sector.

