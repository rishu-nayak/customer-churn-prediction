# ================================
# STEP 1: IMPORT LIBRARIES
# ================================

import pandas as pd          # For data handling
import numpy as np           # For numerical operations

import matplotlib.pyplot as plt   # For plotting
import seaborn as sns             # For visualization

# ================================
# STEP 2: LOAD DATASET
# ================================

# NOTE: Use raw string r"" to avoid path errors in Windows (There is NO \ in path So r is not required)
df = pd.read_csv(r"customer_churn_dataset-training-master.csv")

# Show first 5 rows
print("First 5 rows:\n", df.head())

# Show dataset info
print("\nDataset Info:")
print(df.info())

# ================================
# STEP 3: DATA CLEANING
# ================================

# Remove unnecessary column if exists
if 'customerID' in df.columns: # check column exists ( customerID is unique identifier Does NOT help prediction)
    df.drop('customerID', axis=1, inplace=True) # remove unnecessary column

# Check missing values
print("\nMissing values:\n", df.isnull().sum())

# Drop missing values (simple approach)
df.dropna(inplace=True)

# ================================
# STEP 4: TARGET COLUMN CONVERSION
# ================================

# Convert 'Churn' column from Yes/No to 1/0
# Churn already numeric hai → no need to convert
print(df['Churn'].value_counts())

# ================================
# STEP 5: ENCODE CATEGORICAL DATA
# ================================

# Convert all categorical columns into numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# ================================
# STEP 6: SPLIT FEATURES & TARGET
# ================================

# X = input features, y = output (Churn)
X = df.drop('Churn', axis=1) #Takes all columns except ‘Churn’
y = df['Churn']
#We split the dataset into features (X) and target (y).
#  X contains all input variables used to train the model,
#  while y contains the output variable we want to predict.
#  We remove the target column from X to prevent data leakage.

# ================================
# STEP 7: TRAIN TEST SPLIT
# ================================

from sklearn.model_selection import train_test_split

# Split data into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# STEP 8: FEATURE SCALING
# ================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training data and transform
X_train = scaler.fit_transform(X_train)

# Only transform test data
X_test = scaler.transform(X_test)

# ================================
# STEP 9: TRAIN MODEL
# ================================

from sklearn.ensemble import RandomForestClassifier

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# ================================
# STEP 10: PREDICTION
# ================================

y_pred = model.predict(X_test)

# ================================
# STEP 11: MODEL EVALUATION
# ================================

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Detailed report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================================
# STEP 12: CONFUSION MATRIX
# ================================

cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")  # Save instead of show
# plt.show()

# ================================
# STEP 13: FEATURE IMPORTANCE
# ================================

# Get feature importance from Random Forest
importances = model.feature_importances_

# Feature names
features = X.columns

# Create DataFrame
feat_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop Important Features:\n", feat_df.head())

# ================================
# STEP 14: SAVE MODEL (IMPROVED)
# ================================

import pickle

# Save model, scaler, and columns
pickle.dump(model, open("churn_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

print("\nModel, scaler, and columns saved successfully!")

# ================================
# STEP 15: TEST WITH SAMPLE DATA (FIXED)
# ================================

# Take one sample from ORIGINAL (before scaling)
sample = X.iloc[0:1]   # dataframe format (IMPORTANT)

# Apply SAME scaling
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)

# Output result
if prediction[0] == 1:
    print("\nPrediction: Customer WILL churn")
else:
    print("\nPrediction: Customer will STAY")