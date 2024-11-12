# Analyze customer churn for a telecommunications company to identify high-risk segments and recommend retention strategies

## Overview:

### Step - 01 : Import Necessary Libraries

- **pandas and numpy:** Used for data manipulation and numerical operations.
- **matplotlib.pyplot and seaborn:** Used for data visualization, like plotting graphs.
- **sklearn.model_selection:** Provides functions for train-test split, cross-validation, and grid search for hyperparameter tuning.
- **sklearn.preprocessing:** Contains tools for scaling features and encoding categorical data.
- **sklearn.ensemble and xgboost:** Provides the classifiers for machine learning models (RandomForestClassifier and XGBClassifier).
- **sklearn.metrics:** Provides tools for evaluating model performance, including confusion matrix, ROC-AUC, and classification report.

### Step - 02 : Load and Prepare Data

- Loads a dataset named *'customer_churn_dataset-training-master.csv'* into a **DataFrame** `df`.
- Giving a summary of the DataFrame, including data types and non-null counts.
- Displaying the count of missing values in each column.
- Filling any missing values using forward fill, where each **NaN** value is replaced by the previous row’s value.

### Step - 03 : Encode Categorical Variables

- Defines a list of categorical columns (categorical_cols) that need encoding.
- Uses `LabelEncoder` to convert each column to numeric values. The encoders are saved in `label_encoders` for potential reuse, such as decoding values.

### Step - 04 : Exploratory Data Analysis (EDA)

- Creates a count plot showing the relationship between `Contract Length` and `Churn` status, helping visualize which contract types have higher churn rates.

### Step - 05 : Correlation Heatmap

- Calculates the correlation matrix (`df.corr()`), which quantifies the relationships between numerical features.
- Uses `sns.heatmap()` to visualize correlations, with `annot=True` displaying correlation values on the heatmap and `cmap='coolwarm'` coloring the heatmap.

### Step - 06 : Feature Engineering

- Creates `tenure_group` by segmenting `Tenure` into bins.
- Calculates `monthly_usage` by dividing `Total Spend` by `Tenure + 1` to avoid division by zero.
- Drops `CustomerID` as it’s unnecessary for modeling.

### Step - 07 : Split Data into Train and Test Sets

- Separates features (`X`) and target (`y`). `X` includes all columns except `Churn`, while `y` contains only the `Churn` column.
- Splits the data into training (70%) and testing (30%) sets using `train_test_split`.

### Step - 08 : Process Categorical Columns with One-Hot Encoding

- Converts categorical columns to one-hot encoding, dropping the first category to avoid multicollinearity.
- Ensures `X_train` and `X_test` have matching columns with `.align()`, filling any missing values with 0.

### Step - 09 : Feature Scaling

- Scales the features to zero mean and unit variance, important for certain models and helpful for comparability across features.

### Step - 10 : Model Building and Training

- Initializes an **XGBoost** model with `enable_categorical=True` to handle categorical variables.

### Step - 11 : Hyperparameter Tuning with GridSearchCV

- Sets up a grid of hyperparameters for tuning.
- `GridSearchCV` conducts a grid search with cross-validation to find the optimal parameters, maximizing the ROC-AUC score. The `cv=5` parameter performs 5-fold cross-validation.
- Prints the best parameters and assigns the tuned model to `best_model`.

### Step - 12 : Model Evaluation

- `y_pred` contains predicted labels, and `y_prob` contains predicted probabilities of the positive class.
- Prints a classification report (**precision, recall, F1-score**) and confusion matrix.
- Calculates the **ROC-AUC** score and plots the **ROC curve** to evaluate the classifier's performance visually.

### Step - 13 : Cross-Validation for Reliability

- Uses `cross_val_score` to calculate **ROC-AUC** scores with **5-fold cross-validation** on the full dataset, providing a reliability measure.
- Prints the average **ROC-AUC** score to summarize the model’s performance across folds.

## Tools Used

- **SQL :** For querying and managing large datasets.
- **Python :** For data cleaning, EDA, and model building.
- **scikit-learn & XGBoost :** For machine learning models.


## Goal Accomplishment

1. **Data Preparation:** The code loads, cleans, and prepares data, handling missing values, encoding categorical variables, and creating new features. This ensures that the data is structured and meaningful for analysis.

2. **Exploratory Data Analysis (EDA):** Through visualizations (like count plots and heatmaps), the code examines relationships between variables (e.g., **contract type and churn**), highlighting patterns that may inform the model's predictions.

3. **Feature Engineering:** New features are created to capture customer behavior more effectively, such as `tenure_group` and `monthly_usage`. This adds additional, relevant information that the model can use to understand customer usage patterns.

4. **Modeling and Prediction:** Using XGBoost, a robust algorithm for classification tasks, the code builds a model to predict churn. Hyperparameter tuning is performed to optimize the model’s predictive power.

5. **Model Evaluation:** The code evaluates the model using metrics like **ROC-AUC**, confusion matrix, and cross-validation, which assesses the model’s accuracy and reliability in predicting churn.

6. **Purpose:** Ultimately, this analysis aims to provide the business with a tool to identify at-risk customers. By predicting churn, the organization can proactively target these customers with retention strategies, potentially improving customer satisfaction and reducing revenue losses associated with churn.