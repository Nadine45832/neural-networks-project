# Model 2, Neural Network for Dropout Prediction

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# load dataset
file_path = r"./data/Student data.csv"
df = pd.read_csv(file_path, skiprows=24)

df.columns = [
    "First_Term_GPA",
    "Second_Term_GPA",
    "First_Language",
    "Funding",
    "School",
    "FastTrack",
    "Coop",
    "Residency",
    "Gender",
    "Prev_Education",
    "Age_Group",
    "HS_Average",
    "Math_Score",
    "English_Grade",
    "FirstYearPersistence",
]

# data preprocessing
num_cols = ["First_Term_GPA", "Second_Term_GPA", "HS_Average", "Math_Score"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

# drop irrelevant columns
df = df.drop(columns=["School"])

# missing value handling
df["Prev_Education"] = df["Prev_Education"].replace(["0", "?"], "Unknown")
df["Age_Group"] = df["Age_Group"].replace("?", "Unknown")
df["First_Language"] = df["First_Language"].replace("?", "Unknown")
df["English_Grade"] = df["English_Grade"].replace("?", "Unknown")

# selective imputation, EXCEPT Second_Term_GPA
df["First_Term_GPA"] = df["First_Term_GPA"].fillna(
    df["First_Term_GPA"].median()
)
df["HS_Average"] = df["HS_Average"].fillna(df["HS_Average"].median())
df["Math_Score"] = df["Math_Score"].fillna(df["Math_Score"].median())
# NOTE: Second_Term_GPA left untouched (NaNs preserved)

# dropoutflag definition
df["DropoutFlag"] = df["Second_Term_GPA"].isna().astype(int)

print("DropoutFlag distribution:\n", df["DropoutFlag"].value_counts())

# features and target
columns_to_drop = ["Second_Term_GPA", "DropoutFlag"]
X = df.drop(columns=columns_to_drop, errors="ignore")
y = df["DropoutFlag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# preprocessing pipelines
num_cols = ["First_Term_GPA", "HS_Average", "Math_Score"]
cat_cols = [
    "First_Language",
    "Funding",
    "FastTrack",
    "Coop",
    "Residency",
    "Gender",
    "Prev_Education",
    "Age_Group",
    "English_Grade",
]

num_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

cat_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ]
)

preprocessor.fit(X_train)
X_train_proc = np.asarray(preprocessor.transform(X_train).todense())
X_test_proc = np.asarray(preprocessor.transform(X_test).todense())

print("Preprocessed shapes:")
print("X_train_proc:", X_train_proc.shape)
print("X_test_proc :", X_test_proc.shape)

# neural network architecture
dropout_model = keras.Sequential(
    [
        layers.Dense(
            128, activation="relu", input_shape=(X_train_proc.shape[1],)
        ),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
)

dropout_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

dropout_model.summary()

# train neural network
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(
        "best_dropout_model.keras", save_best_only=True
    ),
]

history = dropout_model.fit(
    X_train_proc,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=2,
)

# evaluate neural network

probs = dropout_model.predict(X_test_proc).ravel()
preds = (probs >= 0.5).astype(int)

acc = accuracy_score(y_test, preds)
print("\nDropout NN Test Accuracy:", acc)

print("\nDropout NN Classification Report:")
print(classification_report(y_test, preds))

cm = confusion_matrix(y_test, preds, labels=[0, 1])
print("\nConfusion Matrix:\n", cm)

# visualizations
""""
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Dropout NN Confusion Matrix")
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

"""
# saving model
dropout_model.save("./model2.keras")
joblib.dump(preprocessor, r"preprocessor_model2.pkl")

print("model saved as dropout_probability_model2.keras")
