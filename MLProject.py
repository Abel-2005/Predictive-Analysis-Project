import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
#pd.set_option('display.max_columns',None)
df = pd.read_csv("C:\\Users\\abelb\\Downloads\\weather.csv")

df.head(10)
df.info()
num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(include="object").columns


df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

print(df.isnull().sum())


df["temperature_celsius"] = df["temperature_celsius"].clip(-30, 50)


plt.figure(figsize=(6,4))
plt.hist(df["temperature_celsius"], bins=30)
plt.title("Temperature Distribution")
plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()


df_reg = df.select_dtypes(include="number")
X_reg = df_reg.drop("temperature_celsius", axis=1)
y_reg = df_reg["temperature_celsius"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler = StandardScaler()
Xr_train = scaler.fit_transform(Xr_train)
Xr_test = scaler.transform(Xr_test)


lr = LinearRegression()
lr.fit(Xr_train, yr_train)
yr_pred_lr = lr.predict(Xr_test)

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(Xr_train, yr_train)
yr_pred_rf = rf.predict(Xr_test)


reg_results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [
        mean_absolute_error(yr_test, yr_pred_lr),
        mean_absolute_error(yr_test, yr_pred_rf)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(yr_test, yr_pred_lr)),
        np.sqrt(mean_squared_error(yr_test, yr_pred_rf))
    ],
    "R2": [
        r2_score(yr_test, yr_pred_lr),
        r2_score(yr_test, yr_pred_rf)
    ]
})

print(reg_results)


plt.figure(figsize=(6,4))
sns.barplot(data=reg_results, x="Model", y="R2")
plt.title("Regression Model Comparison (R²)")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(yr_test, yr_pred_rf, alpha=0.3)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Random Forest: Actual vs Predicted")
plt.show()


df_clf = df[df["condition_text"].value_counts()[df["condition_text"]].values >= 500]
df_clf = df_clf.groupby("condition_text").head(2000)

X_clf = df_clf.select_dtypes(include="number")
y_clf = df_clf["condition_text"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

scaler = StandardScaler()
Xc_train = scaler.fit_transform(Xc_train)
Xc_test = scaler.transform(Xc_test)


log_model = LogisticRegression(max_iter=1000, n_jobs=-1)
log_model.fit(Xc_train, yc_train)
yc_pred_log = log_model.predict(Xc_test)

dt_clf = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_clf.fit(Xc_train, yc_train)
yc_pred_dt = dt_clf.predict(Xc_test)



clf_results = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree"],
    "Accuracy": [
        accuracy_score(yc_test, yc_pred_log),
        accuracy_score(yc_test, yc_pred_dt)
    ],
    "Precision": [
        precision_score(yc_test, yc_pred_log, average="macro"),
        precision_score(yc_test, yc_pred_dt, average="macro")
    ],
    "Recall": [
        recall_score(yc_test, yc_pred_log, average="macro"),
        recall_score(yc_test, yc_pred_dt, average="macro")
    ],
    "F1-score": [
        f1_score(yc_test, yc_pred_log, average="macro"),
        f1_score(yc_test, yc_pred_dt, average="macro")
    ]
})

print(clf_results)


plt.figure(figsize=(6,4))
sns.barplot(data=clf_results, x="Model", y="Accuracy")
plt.title("Classification Accuracy Comparison")
plt.show()


cm = confusion_matrix(yc_test, yc_pred_dt)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
