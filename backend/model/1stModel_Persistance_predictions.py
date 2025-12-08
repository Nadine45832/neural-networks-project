import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.neural_network import MLPRegressor
from imblearn.pipeline import Pipeline 
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

#-----------------------------------------------
#       Data Analyses and Preparation
#------------------------------------------------
# Load the dataset
file_path = r"C:\Users\josep\Downloads\Student_data.csv"

# Skiped the first 24 rows so we only load student records
df = pd.read_csv(file_path, skiprows=24)
print(df.head())
print(df.info())

# Rename columns 
df.columns = [
    'First_Term_GPA','Second_Term_GPA','First_Language','Funding','School',
    'FastTrack','Coop','Residency','Gender','Prev_Education','Age_Group',
    'HS_Average','Math_Score','English_Grade','FirstYearPersistence'
]

# Converted numerical variables to float
num_cols = ['First_Term_GPA','Second_Term_GPA','HS_Average','Math_Score']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

# Initial data exploration
print('Shape of the dataset:')
print(df.shape)
print('Info of the dataset:')
print(df.info())
print(df.columns.tolist())
print(df.describe())
df.columns = df.columns.str.strip().str.replace("'", "").str.replace(" ", "_")
print(df.columns)

# Target variable distribution
print(df['FirstYearPersistence'].value_counts())

# Statistical summary for numerical variables
print(df[['First_Term_GPA','Second_Term_GPA','HS_Average','Math_Score']].describe())
num_cols = ['First_Term_GPA','Second_Term_GPA','HS_Average','Math_Score']
print("\nMissing values in numerical columns:")
print(df[num_cols].isnull().sum())

# Frequencies for categorical variables
categorical_cols = ['First_Language','Funding','School','FastTrack','Coop','Residency','Gender','Prev_Education','Age_Group','English_Grade']
for col in categorical_cols:
    print(f"\n{col} distribution:")
    print(df[col].value_counts())
    

#Data Clean up
#All records belong to one school, so we can drop this column, not relevant for model
df = df.drop(columns=['School'])

#Handling Missing Values
# '0' and '?' as missing values, replacing them with 'Unknown' to avoid misleading the model
df['Prev_Education'] = df['Prev_Education'].replace(['0','?'],'Unknown')
df['Age_Group'] = df['Age_Group'].replace('?', 'Unknown')
df['First_Language'] = df['First_Language'].replace('?', 'Unknown')
df['English_Grade'] = df['English_Grade'].replace('?', 'Unknown')

# For numerical columns, we replaced missing values with median
df['First_Term_GPA'] = df['First_Term_GPA'].fillna(df['First_Term_GPA'].median())
df['Second_Term_GPA'] = df['Second_Term_GPA'].fillna(df['Second_Term_GPA'].median())
df['HS_Average'] = df['HS_Average'].fillna(df['HS_Average'].median())
df['Math_Score'] = df['Math_Score'].fillna(df['Math_Score'].median())
print("\nStats After Cleanup:")

# Frequencies for categorical variables
categorical_cols = ['First_Language','Funding','FastTrack','Coop','Residency','Gender','Prev_Education','Age_Group','English_Grade']
for col in categorical_cols:
    print(f"\n{col} distribution:")
    print(df[col].value_counts())
    
print("\nMissing values in numerical columns:")
print(df[num_cols].isnull().sum())

print('final data types:')
print(df.dtypes)
    
# Isolate features and target variable
X = df.drop(columns=['FirstYearPersistence'])
y = df['FirstYearPersistence']

#Plotting Distributions of Numerical Features
#num_cols = ['First_Term_GPA','Second_Term_GPA','HS_Average','Math_Score']
#df[num_cols].hist(bins=20, figsize=(10,8))
#plt.suptitle("Distributions of Numerical Features")
#plt.show()

#Persistance vs Numerical Features
#for col in num_cols:
    #plt.figure(figsize=(6,4))
    #sns.boxplot(x='FirstYearPersistence', y=col, data=df)
    #plt.title(f"{col} vs Persistence")
    #plt.show()


#Correlation Heatmap
#plt.figure(figsize=(10,8))
#sns.heatmap(df[num_cols + ['FirstYearPersistence']].corr(), annot=True, cmap='coolwarm')
#plt.title("Correlation Heatmap")
#plt.show()

# List of categorical columns
categorical_cols = [
    'First_Language','Funding','FastTrack','Coop','Residency','Gender',
    'Prev_Education','Age_Group','English_Grade'
]

# Chi-Square test for each categorical variable vs persistence
for col in categorical_cols:
    contingency = pd.crosstab(df[col], df['FirstYearPersistence'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"{col}: Chi2 = {chi2:.2f}, p-value = {p:.4f}")


#-----------------------------------------------
#       Data  Preparation
#------------------------------------------------
print("\nData Preprocessing:")

# Numerical and Categorical columns
num_cols = ['First_Term_GPA','Second_Term_GPA','HS_Average','Math_Score']
cat_cols = ['First_Language','Funding','FastTrack','Coop','Residency','Gender',
            'Prev_Education','Age_Group','English_Grade']

# Preprocessing for numerical data,  median imputation and scaling
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data, constant imputation and one-hot encoding
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

#-----------------------------------------------
#  Building the model
#------------------------------------------------


#-----------------------------------------------
#    1st Model MLP classifier with grid search
#------------------------------------------------
print("\nModel 1: MLP Classifier with Grid Search")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# MLP classifier with grid search
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(
        solver='adam',
        random_state=42,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.15,
        max_iter=500
    ))
])

param_grid = {
    'classifier__hidden_layer_sizes': [
        (64, 32),
        (128, 32),
        (32, 16)
    ],
    'classifier__activation': ['relu', 'tanh'],
    'classifier__alpha': [0.0001, 0.001, 0.01],
    'classifier__learning_rate_init': [0.0005, 0.001, 0.005],
}

grid = GridSearchCV(
    clf,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)

y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC, Model 1 (MLP):", roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1]))

#-----------------------------------------------
#    2nd Model hybrid random forest leaf features + neural network
#------------------------------------------------

print("\nModel 2: hybrid random forest leaf features + neural network")

X = df.drop(columns=["FirstYearPersistence"])
y = df["FirstYearPersistence"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# used dense arrays
X_train_proc = np.asarray(X_train_proc.todense())
X_test_proc = np.asarray(X_test_proc.todense())

print("Preprocessed shapes:")
print("X_train_proc:", X_train_proc.shape)
print("X_test_proc :", X_test_proc.shape)

# random forrest training on features

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_proc, y_train)

# leafs indices
X_train_leaves = rf.apply(X_train_proc)
X_test_leaves = rf.apply(X_test_proc)

print("Leaf indices shape (train):", X_train_leaves.shape)

# one hot encoding for leaf indices
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_leaves_enc = encoder.fit_transform(X_train_leaves)
X_test_leaves_enc = encoder.transform(X_test_leaves)
print("Encoded leaf features shape (train):", X_train_leaves_enc.shape)

# Combine original preprocessed features with leaf features
X_train_nn = np.hstack([X_train_proc, X_train_leaves_enc])
X_test_nn = np.hstack([X_test_proc, X_test_leaves_enc])
print("Final NN input shape (train):", X_train_nn.shape)


# neural network 

nn_model = keras.Sequential([
    layers.Dense(256, activation="relu", input_shape=(X_train_nn.shape[1],)),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

nn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

nn_model.summary()

# training neural network

callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_rf_nn_model.keras", save_best_only=True)
]

history = nn_model.fit(
    X_train_nn, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=2
)

#  evaluation

probs = nn_model.predict(X_test_nn).ravel()
preds = (probs >= 0.5).astype(int)

acc = accuracy_score(y_test, preds)
print("\nRF+NN Test Accuracy:", acc)

print("\nRF+NN Classification Report:")
print(classification_report(y_test, preds))

cm = confusion_matrix(y_test, preds)
print("\nConfusion Matrix:\n", cm)

roc_auc = roc_auc_score(y_test, probs)
print("\nRF+NN ROC-AUC:", roc_auc)


# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("RF+NN ROC Curve")
plt.legend()
plt.show()

import joblib

# exporting random forest model
joblib.dump(rf, r"C:\Users\josep\Downloads\rf_leaf_model.pkl")

# exporting leaf encoder
joblib.dump(encoder, r"C:\Users\josep\Downloads\rf_leaf_encoder.pkl")

# exporting preprocessor (ColumnTransformer)
joblib.dump(preprocessor, r"C:\Users\josep\Downloads\preprocessor.pkl")

# exporting NN model
nn_model.save(r"C:\Users\josep\Downloads\student_persistence_hybrid_rf_nn.keras")

print("Hybrid RF+NN components saved: rf_leaf_model.pkl, rf_leaf_encoder.pkl, preprocessor.pkl, student_persistence_hybrid_rf_nn.keras")