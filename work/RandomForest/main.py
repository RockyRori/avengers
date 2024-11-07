import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

"""
use embedded model RandomForestClassifier in sklearn to predict answers
"""
if __name__ == '__main__':
    pass

"""
preprocessing
"""
# read data from training set
train = pd.read_csv('train.csv')
print(train.head())

# read data from testing set
test = pd.read_csv('test.csv')
print(test.head())

# display attributes and distinct values count
train.nunique().sort_values(ascending=False)
train.info()

# remove useless attributes
train = train.drop(['id'], axis=1)
train.head()

# remain id in test to match predictions
# test = test.drop(['id'], axis=1)
test.head()

# calculate distribution of NObeyesdad values
print(round(train['NObeyesdad'].value_counts() * 100 / len(train), 2))

# select object typed attributes except target
cat = list(train.select_dtypes(['object']).columns)
cat.remove('NObeyesdad')

# Checks whether the class features in the training set and the test set are consistent
# if not then remind
for i in cat:
    if len(list(set(train[i].unique().tolist()) ^ set(test[i].unique().tolist()))) != 0:
        print(i, 'needs attention')
    else:
        continue

# only CALC is not aligned
print(train['CALC'].value_counts())
print(test['CALC'].value_counts())

# replace new appearance with shown
test['CALC'] = test['CALC'].replace('Always', 'Frequently')

"""
modeling
"""
# Step 1: Encode categorical variables in the training set
train_encoded = pd.get_dummies(train,
                               columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC',
                                        'CALC', 'MTRANS'])
# Assuming 'NObeyesdad' is the target variable
Y_train = train_encoded['NObeyesdad']
# remove 'NObeyesdad' from training set
X_train = train_encoded.drop(['NObeyesdad'], axis=1)

# Step 2: Train the model
model = RandomForestClassifier(random_state=28)
model.fit(X_train, Y_train)

# Step 3: Encode categorical variables in the test set
test_encoded = pd.get_dummies(test, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC',
                                             'CALC', 'MTRANS'])

# Step 4: Ensure the features in the test set match those used during training
missing_features = set(X_train.columns) - set(test_encoded.columns)
# Add missing features with zeros
for feature in missing_features:
    test_encoded[feature] = 0

# Step 5: Reorder columns in the test set to match the order in the training set
test_encoded = test_encoded[X_train.columns]

# Step 6: Generate Predictions
predictions = model.predict(test_encoded)

# Step 7: Create Submission DataFrame
submission_df = pd.DataFrame({'id': test['id'], 'NObeyesdad': predictions})

# Step 8: Save to CSV
submission_df.to_csv('submission.csv', index=False)

"""
validating
"""
# Step 1: Split your data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=28)

# Step 2: Model training
model = RandomForestClassifier(random_state=28)
model.fit(X_train, Y_train)

# Step 3: Generate Predictions on the validation set
val_predictions = model.predict(X_val)

# Step 4: Check importance
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
print(feature_importances.head(10))

# Step 5: Check Accuracy on the validation set
val_accuracy = accuracy_score(Y_val, val_predictions)
print(f"Validation Accuracy: {val_accuracy}")
