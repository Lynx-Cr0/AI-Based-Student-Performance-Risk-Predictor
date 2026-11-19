"""Train the student performance ANN and baseline. Run: python src/train.py"""
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import json, joblib, numpy as np, pandas as pd, keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"data/student_performance_dataset.csv")
target="performance_category"
features=[c for c in df.columns if c not in ["student_id",target]]
X=df[features]
le=LabelEncoder(); y=le.fit_transform(df[target].astype(str))
X_train_full,X_test,y_train_full,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
X_train,X_val,y_train,y_val=train_test_split(X_train_full,y_train_full,test_size=.20,random_state=42,stratify=y_train_full)
imputer=SimpleImputer(strategy="median"); scaler=StandardScaler()
Xtr=scaler.fit_transform(imputer.fit_transform(X_train))
Xva=scaler.transform(imputer.transform(X_val)); Xte=scaler.transform(imputer.transform(X_test))
Xfull=scaler.transform(imputer.transform(X_train_full))
baseline=LogisticRegression(max_iter=2000,random_state=42).fit(Xfull,y_train_full)
keras.utils.set_random_seed(42)
model=keras.Sequential([keras.Input(shape=(len(features),)),layers.Dense(32,activation="relu"),
 layers.Dropout(.15),layers.Dense(16,activation="relu"),layers.Dense(len(le.classes_),activation="softmax")])
model.compile(optimizer=keras.optimizers.Adam(.001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
history=model.fit(Xtr,y_train,validation_data=(Xva,y_val),epochs=80,batch_size=256,
 callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)],verbose=1)
pred=model.predict(Xte,verbose=0).argmax(axis=1)
base_pred=baseline.predict(Xte)
print("ANN accuracy:",accuracy_score(y_test,pred))
print(classification_report(y_test,pred,target_names=le.classes_))
(ROOT/"models").mkdir(exist_ok=True)
model.save(ROOT/"models/student_performance.keras")
joblib.dump({"imputer":imputer,"scaler":scaler,"label_encoder":le,"feature_columns":features},ROOT/"models/preprocessing.joblib")
joblib.dump(baseline,ROOT/"models/logistic_regression.joblib")
