import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('allhypo.data', delimiter=',')  # Adjust delimiter if needed

# Clean column names
df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces from column names
df.columns = [col.replace('?', 'Unknown').strip() for col in df.columns]

# Handle non-numeric data
df.replace('?', pd.NA, inplace=True)
df.dropna(subset=['SVHC'], inplace=True)  # Replace 'SVHC' with actual target column name

# Define features and target
X = df.drop(columns=['SVHC'])
y = df['SVHC']

# Convert categorical features to numeric
X = pd.get_dummies(X, drop_first=True)

# Convert boolean columns to integers
bool_cols = X.select_dtypes(include='bool').columns
for col in bool_cols:
    X[col] = X[col].astype(int)

# Impute missing values
imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(classification_report(y_test, y_pred_rf))

# Cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"Random Forest Cross-validation scores: {cv_scores}")
print(f"Random Forest Mean CV Accuracy: {cv_scores.mean()}")

# Grid search for hyperparameter tuning (Random Forest)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best parameters for Random Forest: {grid_search.best_params_}")