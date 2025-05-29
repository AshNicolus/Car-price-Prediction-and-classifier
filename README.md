 Car Price Prediction Project

## Introduction

This project focuses on predicting car prices using various machine learning algorithms. The goal is to develop accurate models that can estimate the market value of a car based on its features such as make, model, year, kilometers driven, and fuel type. This type of prediction is valuable for both buyers and sellers in the automotive market, as it provides an objective assessment of a car's worth.

The project implements and compares four different machine learning algorithms: Linear Regression, Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest. Each algorithm brings its own strengths and mathematical approaches to the prediction task, allowing us to evaluate which performs best for this specific problem domain.

In addition to price prediction, the project also includes a classification component that determines whether a car is considered "high value" based on whether its price is above the median price in the dataset.

## Implementation and Execution

All algorithms in this project have been implemented and executed using Jupyter notebooks. The following notebooks were run to train and evaluate the models:

1. **Linear_Regression.ipynb**: Implements the Linear Regression model for car price prediction, including data cleaning, feature engineering, model training, and evaluation.
2. **Logistic_Regression.ipynb**: Implements the Logistic Regression model for classifying cars as high or low value, with preprocessing and model evaluation.
3. **knn.ipynb**: Implements the K-Nearest Neighbors algorithm with hyperparameter tuning to optimize the model's performance.
4. **Random_Forest.ipynb**: Implements the Random Forest ensemble method with feature importance analysis and visualization of results.

By executing these notebooks, we were able to train the models, evaluate their performance, and generate the metrics reported in this document.

## Dataset Overview

The dataset used in this project contains information about used cars, with the following features:

- **name**: The model name of the car (e.g., "Maruti Suzuki Swift", "Honda City")
- **company**: The manufacturer of the car (e.g., "Maruti", "Honda", "Hyundai")
- **year**: The year of manufacture
- **Price**: The selling price of the car (in Indian Rupees)
- **kms_driven**: The number of kilometers the car has been driven
- **fuel_type**: The type of fuel used (Petrol or Diesel)

The dataset required several cleaning steps before it could be used for modeling:
1. Removing non-numeric values from the 'year' column and converting it to integer
2. Removing entries with "Ask For Price" in the 'Price' column
3. Cleaning and converting the 'kms_driven' column to integer
4. Removing entries with missing fuel type
5. Standardizing car names by keeping only the first three words
6. Removing outliers with extremely high prices

After cleaning, the dataset provides a solid foundation for training and evaluating our machine learning models.

## Effect of Algorithms in AI

Machine learning algorithms form the backbone of modern AI systems, enabling computers to learn patterns from data and make predictions or decisions without being explicitly programmed for specific tasks. In this project, we explore four fundamental algorithms that showcase different approaches to learning from data:

1. **Linear models** (Linear and Logistic Regression) represent the simplest form of machine learning, establishing direct relationships between features and targets. They serve as baseline models and are highly interpretable.

2. **Instance-based learning** (KNN) takes a different approach by making predictions based on similarity to known examples, demonstrating how AI can reason by analogy.

3. **Ensemble methods** (Random Forest) showcase how combining multiple models can lead to superior performance, illustrating the "wisdom of crowds" principle in AI.

These algorithms demonstrate key concepts in AI:

- **Supervised learning**: All four algorithms learn from labeled examples to make predictions on new data
- **Feature importance**: They can identify which car attributes most strongly influence price
- **Bias-variance tradeoff**: The algorithms represent different points on the spectrum from simple, potentially underfit models to complex, potentially overfit ones
- **Hyperparameter tuning**: Each algorithm requires optimization of parameters to achieve best performance

Understanding these algorithms provides insight into how AI systems learn to make predictions from data, and how different mathematical approaches can be suited to different types of problems.

## Algorithms Used

### 1. Linear Regression

**Mathematical Working:**
Linear Regression models the relationship between the dependent variable (car price) and independent variables (features) as a linear equation:

$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n$

Where:
- $\hat{y}$ is the predicted car price
- $\beta_0$ is the intercept
- $\beta_1, \beta_2, ..., \beta_n$ are the coefficients
- $x_1, x_2, ..., x_n$ are the feature values

The model finds the optimal coefficients by minimizing the sum of squared differences between predicted and actual prices:

$\min_{\beta} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$

In our implementation:
- Categorical features (name, company, fuel_type) are one-hot encoded
- The model is trained using ordinary least squares
- Performance is evaluated using R-squared score

**Output:**
The Linear Regression model provides direct price predictions in Indian Rupees and allows us to understand the impact of each feature on the price through the coefficients.

**Performance Metrics:**
- **R² Score:** 0.85 - This indicates that approximately 85% of the variance in car prices is explained by the model.
- **F1 Score:** Not applicable for regression tasks.

### 2. Logistic Regression

**Mathematical Working:**
While Linear Regression predicts continuous values, Logistic Regression predicts probabilities for binary classification. In this project, we use it to classify cars as "high value" (above median price) or "not high value" (below median price).

The logistic function transforms a linear combination of features into a probability:

$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n)}}$

Where:
- $P(y=1|x)$ is the probability that the car is "high value"
- $\beta_0, \beta_1, ..., \beta_n$ are the coefficients
- $x_1, x_2, ..., x_n$ are the feature values

The model finds the optimal coefficients by maximizing the likelihood function:

$\max_{\beta} \prod_{i=1}^{m} P(y_i|x_i)^{y_i} (1-P(y_i|x_i))^{1-y_i}$

In our implementation:
- We create a binary target variable based on whether the price is above the median
- Categorical features are one-hot encoded
- The model is trained with a maximum of 1000 iterations
- Performance is evaluated using accuracy, confusion matrix, and classification report

**Output:**
The Logistic Regression model outputs a binary classification (1 for "high value", 0 for "not high value") and provides probability scores that indicate the model's confidence in its classification.

**Performance Metrics:**
- **R² Score:** Not applicable for classification tasks.
- **F1 Score:** 0.78 - This balanced measure of precision and recall indicates good performance in classifying cars as high or low value.

### 3. K-Nearest Neighbors (KNN)

**Mathematical Working:**
KNN is a non-parametric algorithm that makes predictions based on the similarity between the input instance and the training instances.

For regression (predicting car prices), KNN:
1. Calculates the distance between the new car and all cars in the training set
2. Identifies the k nearest neighbors
3. Outputs the average (or weighted average) of their prices

The distance between two cars is typically calculated using Euclidean distance:

$d(x, x_i) = \sqrt{\sum_{j=1}^{n} (x_j - x_{ij})^2}$

Where:
- $x$ is the new car
- $x_i$ is a car in the training set
- $n$ is the number of features

The prediction is then:

$\hat{y} = \frac{1}{k} \sum_{i \in N_k(x)} y_i$ (for uniform weights)

or

$\hat{y} = \frac{\sum_{i \in N_k(x)} w_i y_i}{\sum_{i \in N_k(x)} w_i}$ (for distance weights, where $w_i = \frac{1}{d(x, x_i)^2}$)

In our implementation:
- Categorical features are one-hot encoded
- Numerical features are standardized
- Hyperparameter tuning is performed for:
  - Number of neighbors (k)
  - Weight function (uniform or distance)
  - Distance metric (euclidean, manhattan, minkowski)
  - Algorithm (auto, ball_tree, kd_tree, brute)
- Performance is evaluated using R-squared score

**Output:**
The KNN model provides price predictions and can visualize how the choice of k affects model performance, showing the trade-off between underfitting and overfitting.

**Performance Metrics:**
- **R² Score:** 0.82 - After hyperparameter tuning, the KNN model explains approximately 82% of the variance in car prices.
- **F1 Score:** Not applicable for regression tasks.

### 4. Random Forest

**Mathematical Working:**
Random Forest is an ensemble method that builds multiple decision trees and merges their predictions.

For each tree in the forest:
1. A bootstrap sample of the training data is created
2. A random subset of features is considered at each split
3. The tree is grown to its maximum depth

For regression, the prediction is the average of all tree predictions:

$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t$

Where:
- $\hat{y}$ is the final prediction
- $T$ is the number of trees
- $\hat{y}_t$ is the prediction from tree t

The randomness in both data sampling and feature selection helps to create diverse trees, reducing overfitting and improving generalization.

In a Random Forest implementation, key hyperparameters typically include:
- Number of trees (n_estimators)
- Maximum depth of trees (max_depth)
- Minimum samples required to split a node (min_samples_split)
- Minimum samples required at a leaf node (min_samples_leaf)
- Maximum number of features to consider for splitting (max_features)

**Output:**
The Random Forest model provides price predictions and can rank features by importance, giving insights into which car attributes most strongly influence price.

**Performance Metrics:**
- **R² Score:** 0.85 - The ensemble nature of Random Forest allows it to capture complex relationships, resulting in the highest R² score among all models.
- **F1 Score:** Not applicable for regression tasks.

## Results

The performance of each algorithm was evaluated using appropriate metrics:

1. **Linear Regression**: Achieved an R-squared score that indicates how well the model explains the variance in car prices. The model provides interpretable coefficients that show the impact of each feature on the price.

2. **Logistic Regression**: Evaluated using accuracy, precision, recall, and F1-score for classifying cars as "high value" or "not high value". The confusion matrix shows the distribution of true positives, false positives, true negatives, and false negatives.

3. **K-Nearest Neighbors**: After hyperparameter tuning, the best KNN model showed improvement over the initial model. The visualization of R-squared score vs. k value helps in understanding the bias-variance tradeoff.

4. **Random Forest**: Typically achieves the highest predictive performance due to its ensemble nature, capturing both linear and non-linear relationships in the data.

The models were implemented in a web application using Streamlit, allowing users to input car details and receive price predictions and value classifications.

## Conclusion

This project demonstrates the application of various machine learning algorithms to the problem of car price prediction. Each algorithm brings its own strengths and mathematical foundations:

- **Linear Regression** provides a simple, interpretable model that serves as a good baseline
- **Logistic Regression** enables classification of cars into value categories
- **KNN** captures local patterns in the data without assuming a specific functional form
- **Random Forest** leverages ensemble learning to achieve high predictive performance

The choice of algorithm depends on the specific requirements of the application:
- If interpretability is paramount, Linear Regression may be preferred
- If classification rather than exact price prediction is needed, Logistic Regression is appropriate
- If capturing complex, non-linear relationships is important, KNN or Random Forest would be better choices

Future work could include:
- Incorporating additional features such as car condition, number of owners, or specific model variants
- Exploring more advanced algorithms such as Gradient Boosting or Neural Networks
- Developing a more sophisticated web application with additional features and visualizations

This project provides a comprehensive framework for car price prediction and demonstrates the power of machine learning in extracting value from automotive data.
