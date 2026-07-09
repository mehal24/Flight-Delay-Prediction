import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_csv(r"C:\Users\mehal\OneDrive\Desktop\SEM-6\DWDM\advanced_flight_delays_1M.csv.xls")

# Reduce dataset size for faster training
df = df.sample(100000, random_state=42)

print(df.head())
print(df.info())


# -----------------------------
# CREATE TARGET VARIABLE
# -----------------------------

df["Delayed"] = df["Delay_Minutes"].apply(lambda x: 1 if x > 15 else 0)


# -----------------------------
# DELAY SEVERITY PIE CHART
# -----------------------------

def delay_severity(x):

    if x <= 15:
        return "On Time"
    elif x <= 60:
        return "Short Delay"
    elif x <= 180:
        return "Medium Delay"
    else:
        return "Severe Delay"

df["DelayCategory"] = df["Delay_Minutes"].apply(delay_severity)

plt.figure(figsize=(7,7))

df["DelayCategory"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Distribution of Flight Delays by Severity")
plt.ylabel("")
plt.show()


# -----------------------------
# FEATURE CORRELATION HEATMAP
# -----------------------------

plt.figure(figsize=(10,7))

corr = df.corr(numeric_only=True)

sns.heatmap(corr, cmap="coolwarm")

plt.title("Feature Correlation Heatmap")

plt.show()


# -----------------------------
# AIRLINE RISK RANKING
# -----------------------------

airline_risk = df.groupby("Airline")["Delayed"].mean().sort_values(ascending=False)

plt.figure(figsize=(8,5))

airline_risk.plot(kind="bar")

plt.title("Airline Delay Risk Ranking")
plt.ylabel("Delay Probability")

plt.show()


# -----------------------------
# ORIGIN AIRPORT RISK
# -----------------------------

origin_risk = df.groupby("Origin")["Delayed"].mean().sort_values(ascending=False)

plt.figure(figsize=(8,5))

origin_risk.head(10).plot(kind="bar")

plt.title("Origin Airport Delay Risk")

plt.show()


# -----------------------------
# DESTINATION AIRPORT RISK
# -----------------------------

dest_risk = df.groupby("Destination")["Delayed"].mean().sort_values(ascending=False)

plt.figure(figsize=(8,5))

dest_risk.head(10).plot(kind="bar")

plt.title("Destination Airport Delay Risk")

plt.show()


# -----------------------------
# ENCODE CATEGORICAL VARIABLES
# -----------------------------

le = LabelEncoder()

df["Airline"] = le.fit_transform(df["Airline"])
df["Origin"] = le.fit_transform(df["Origin"])
df["Destination"] = le.fit_transform(df["Destination"])
df["Maintenance_Status"] = le.fit_transform(df["Maintenance_Status"])


# -----------------------------
# FEATURE SELECTION
# -----------------------------

features = [

"Airline",
"Origin",
"Destination",
"Passenger_Count",
"Precipitation_Forecast_mm",
"Wind_Speed_Forecast_mph",
"Airport_Traffic_Index",
"Inbound_Flight_Delayed"

]

X = df[features]

y = df["Delayed"]


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(

X,
y,
test_size=0.2,
random_state=42

)


# -----------------------------
# TRAIN MULTIPLE MODELS
# -----------------------------

print("Training models...")

models = {

"Logistic Regression": LogisticRegression(max_iter=500, solver="liblinear"),

"Decision Tree": DecisionTreeClassifier(),

"Random Forest": RandomForestClassifier(n_estimators=100),

"Gradient Boosting": GradientBoostingClassifier()

}

results = {}

for name, model in models.items():

    print("Training:", name)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print(name, "Accuracy:", acc)

    results[name] = acc


# -----------------------------
# ACCURACY COMPARISON GRAPH
# -----------------------------

plt.figure(figsize=(8,5))

plt.bar(results.keys(), results.values())

plt.title("Accuracy Comparison Across Models")

plt.ylabel("Accuracy")

plt.show()


# -----------------------------
# RANDOM FOREST MODEL
# -----------------------------

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:,1]


# -----------------------------
# ROC CURVE
# -----------------------------

fpr, tpr, thresholds = roc_curve(y_test, probs)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))

plt.plot(fpr, tpr, label="AUC = %0.3f" % roc_auc)

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()


# -----------------------------
# PRECISION RECALL CURVE
# -----------------------------

precision, recall, thresholds = precision_recall_curve(y_test, probs)

plt.figure(figsize=(7,6))

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.show()


# -----------------------------
# THRESHOLD TUNING GRAPH
# -----------------------------

plt.figure(figsize=(8,6))

plt.plot(thresholds, precision[:-1], label="Precision")

plt.plot(thresholds, recall[:-1], label="Recall")

plt.xlabel("Threshold")

plt.ylabel("Score")

plt.title("Precision vs Recall Tradeoff")

plt.legend()

plt.show()


# -----------------------------
# CONFUSION MATRIX
# -----------------------------

pred = model.predict(X_test)

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()






#Feature importance

# Get importance values
importances = model.feature_importances_

# Create dataframe
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
})

# Sort values
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=True)

# Plot
plt.figure(figsize=(8,5))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])

plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.show()