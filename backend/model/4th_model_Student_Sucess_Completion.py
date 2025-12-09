# Model 4, Neural Network for Predicting Student Success in Program (Completion)

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

# load data

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

df = df.drop(columns=["School"])

num_cols = ["First_Term_GPA", "Second_Term_GPA", "HS_Average", "Math_Score"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

df["Prev_Education"] = df["Prev_Education"].replace(["0", "?"], "Unknown")
df["Age_Group"] = df["Age_Group"].replace("?", "Unknown")
df["First_Language"] = df["First_Language"].replace("?", "Unknown")
df["English_Grade"] = df["English_Grade"].replace("?", "Unknown")


# sucess definition
# dropoutFlag= 1 if Second Term GPA missing
df["DropoutFlag"] = df["Second_Term_GPA"].isna().astype(int)

# success = 1 if not dropout, else 0
df["ProgramSuccess"] = (df["DropoutFlag"] == 0).astype(int)

print("ProgramSuccess distribution:\n", df["ProgramSuccess"].value_counts())


# features and target
X = df.drop(columns=["Second_Term_GPA", "DropoutFlag", "ProgramSuccess"])
y = df["ProgramSuccess"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# preprocessing
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
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ]
)

preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

print("Preprocessed shapes:")
print("X_train_proc:", X_train_proc.shape)
print("X_test_proc :", X_test_proc.shape)


# building model
success_model = keras.Sequential(
    [
        layers.Dense(
            128, activation="relu", input_shape=(X_train_proc.shape[1],)
        ),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

success_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

success_model.summary()

# training
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(
        "best_success_model.keras", save_best_only=True
    ),
]

history = success_model.fit(
    X_train_proc,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=2,
)

# evaluation of NN
probs = success_model.predict(X_test_proc).ravel()
preds = (probs >= 0.5).astype(int)

acc = accuracy_score(y_test, preds)
print("\nProgram Success NN Test Accuracy:", acc)

print("\nProgram Success NN Classification Report:")
print(classification_report(y_test, preds))

cm = confusion_matrix(y_test, preds, labels=[0, 1])
print("\nConfusion Matrix:\n", cm)

# saving model
success_model.save("./model4.keras")
joblib.dump(preprocessor, r"preprocessor_model4.pkl")

print("Program Success model saved as program_success_model4.keras")
