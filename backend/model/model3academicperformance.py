
# Model 3, Academic Performance Regression NN

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers

# load data
file_path = r"C:\Users\josep\Downloads\Student_data.csv"
df = pd.read_csv(file_path, skiprows=24)

df.columns = [
    'First_Term_GPA','Second_Term_GPA','First_Language','Funding','School',
    'FastTrack','Coop','Residency','Gender','Prev_Education','Age_Group',
    'HS_Average','Math_Score','English_Grade','FirstYearPersistence'
]

df = df.drop(columns=['School'])

num_cols = ['First_Term_GPA','Second_Term_GPA','HS_Average','Math_Score']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

df['Prev_Education'] = df['Prev_Education'].replace(['0','?'],'Unknown')
df['Age_Group'] = df['Age_Group'].replace('?', 'Unknown')
df['First_Language'] = df['First_Language'].replace('?', 'Unknown')
df['English_Grade'] = df['English_Grade'].replace('?', 'Unknown')

# feature engineering
df['GPA_Delta'] = df['HS_Average'] - df['First_Term_GPA']
df['Low_GPA'] = (df['First_Term_GPA'] < 2.0).astype(int)
df['High_HS'] = (df['HS_Average'] > 80).astype(int)

# definig target and features
df_reg = df.dropna(subset=['Second_Term_GPA'])

X = df_reg.drop(columns=['Second_Term_GPA'])
y = df_reg['Second_Term_GPA']

# scaling
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y.values.reshape(-1,1)).ravel()

# data splitting
X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(
    X, y_scaled, test_size=0.2, random_state=42
)

# preprocessing pipelines
num_cols = ['First_Term_GPA','HS_Average','Math_Score','GPA_Delta']
cat_cols = ['First_Language','Funding','FastTrack','Coop','Residency','Gender',
            'Prev_Education','Age_Group','English_Grade']
bin_cols = ['Low_GPA','High_HS']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols),
        ('bin', 'passthrough', bin_cols)
    ]
)

preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)



reg_model = keras.Sequential([
    layers.Dense(256, activation="relu", input_shape=(X_train_proc.shape[1],),
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])

reg_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss=keras.losses.Huber(),
    metrics=["mae"]
)

# trainig 
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_reg_model.keras", save_best_only=True)
]

history = reg_model.fit(
    X_train_proc, y_train_scaled,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=2
)


# evaluation of NN
preds_scaled = reg_model.predict(X_test_proc).ravel()
preds = target_scaler.inverse_transform(preds_scaled.reshape(-1,1)).ravel()
y_test = target_scaler.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\nEnhanced Academic Performance NN Results:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)


#  plots
"""
# training vs validation loss
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss (Huber)')
plt.plot(history.history['val_loss'], label='Val Loss (Huber)')
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

# predicted vs actual GPA
plt.figure(figsize=(6,6))
plt.scatter(y_test, preds, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Second Term GPA")
plt.ylabel("Predicted Second Term GPA")
plt.title("Predicted vs Actual GPA")
plt.show()

# residuals
residuals = y_test - preds
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Prediction Error (Actual - Predicted GPA)")
plt.show()

# error vs GPA
plt.figure(figsize=(6,4))
plt.scatter(y_test, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Actual GPA")
plt.ylabel("Residual (Error)")
plt.title("Prediction Error vs Actual GPA")
plt.show()
"""

# comparison between actual gpa and predicted gpa

#example_idx = np.random.choice(len(y_test), size=10, replace=False)
#example_actual = y_test[example_idx]
#example_pred = preds[example_idx]
#print("\nSample Predictions (Actual vs Predicted GPA):")
#for i in range(len(example_idx)):
#    print(f"Student {i+1}: Actual GPA = {example_actual[i]:.2f}, Predicted GPA = {example_pred[i]:.2f}, Error = {example_actual[i]-example_pred[i]:.2f}")

# saving model
reg_model.save(r"C:\Users\josep\Downloads\academic_performance_model3.keras")

print("Academic Performance model saved as academic_performance_model3.keras")