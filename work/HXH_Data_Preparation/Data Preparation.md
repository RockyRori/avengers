### Data Preparation for Multi-Class Prediction of Obesity Risk

#### **Introduction**
The goal of our task is using various health and lifestyle factors to predict a person's obesity risk category, represented by the target variable `NObeyesdad`. The task involves building a multi-class classification model to predict the obesity risk levels for the test dataset. This report outlines the data preparation process used to create a robust machine learning model based on the provided training and test datasets.

---

#### **1. Data Overview**

**Training Data (`train.csv`)**:
- Contains 20,800 rows and 18 columns, including various features such as:
  - **Categorical Features**: `Gender`, `family_history_with_overweight`, `FAVC`, `CAEC`, `SMOKE`, etc.
  - **Numerical Features**: `Age`, `Height`, `Weight`, `FCVC`, `NCP`, etc.
  - **Target Variable**: `NObeyesdad`, representing multiple obesity-related categories, such as `Normal_Weight`, `Obesity_Type_I`, `Overweight_Level_I`, etc.

**Test Data (`test.csv`)**:
- Contains 13,800 rows and 17 columns, similar to the training data but without the target variable `NObeyesdad`.

**Submission File (`sample_submission.csv`)**:
- Provides the format for submission, requiring predictions for `NObeyesdad` for each `id` in the test data.

---

#### **2. Data Cleaning and Preprocessing**

**2.1. Dropping Irrelevant Columns**
- The `id` column, which is a unique identifier, was dropped from the training data as it does not contribute to the predictive modeling process.

  ```python
  train = train.drop(['id'], axis=1)
  ```

  - The `id` column was retained in the test dataset to be used in the final submission for matching predictions with the correct rows.

**2.2. Handling Categorical Variables**
- Categorical variables were encoded using **one-hot encoding** to convert them into numerical format, as required by machine learning algorithms like Random Forest. This was done for both the training and test datasets.

  ```python
  train_encoded = pd.get_dummies(train, 
                                columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'])
  ```

- The target variable `NObeyesdad` was excluded from one-hot encoding and was stored separately as the dependent variable for model training.

**2.3. Ensuring Consistency Between Train and Test Sets**
- A check was performed to ensure that the categorical features in both the train and test datasets were consistent. If discrepancies were found (e.g., differences in the levels of categorical variables), they were addressed before modeling:
  
  - Example: The `CALC` feature in the test set had a new value ("Always") not present in the training set. This was replaced with an already existing category ("Frequently").

  ```python
  test['CALC'] = test['CALC'].replace('Always', 'Frequently')
  ```

**2.4. Handling Missing Features in Test Data**
- After one-hot encoding, There were cases where some features present in the training data were missing in the test set. These missing features were added to the test set with a value of 0 to ensure consistency between the datasets.
  
  ```python
  missing_features = set(X_train.columns) - set(test_encoded.columns)
  for feature in missing_features:
      test_encoded[feature] = 0
  ```

---

#### **3. Feature Engineering**
**3.1. Feature Selection**
- The features were carefully aligned between the training and test data after one-hot encoding. The target variable `NObeyesdad` was separated from the training data as `Y_train` and the remaining features were used as input variables (`X_train`).

- **Potential for further feature engineering**: Calculating **BMI** using `Height` and `Weight` could be an additional feature that directly relates to obesity risk.

---

#### **4. Validation and Model Training**
**4.1. Train-Validation Split**
- The training data was split into a training and a validation set using an 80/20 split to evaluate model performance before making predictions on the test set. This allowed for testing the model’s generalizability and adjusting hyperparameters before making final predictions.

  ```python
  X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=28)
  ```

**4.2. Feature Importance**
- Feature importance was evaluated to understand which features contributed the most to the predictions after training the model. This could provide insights for further feature selection or engineering.

  ```python
  feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
  ```

---

#### **5. Submission File Preparation**
Once the model was trained and predictions were made on the test set, the results were compiled into a submission CSV file. The submission file included two columns:
- `id`: The unique identifier from the test set.
- `NObeyesdad`: The predicted obesity class for each row.

The file was saved and formatted as required by the competition.

```python
submission_df = pd.DataFrame({'id': test['id'], 'NObeyesdad': predictions})
submission_df.to_csv('submission.csv', index=False)
```

---

#### **Conclusion**
The data preparation phase involves several important steps, such as cleaning, encoding categorical variables, handling the differences between the training and test sets, and preparing the data for modeling. Ensuring consistent feature alignment across datasets is critical to avoid errors in the model prediction process. In addition, the importance and validation of features are essential elements in refining the model before making final predictions.
