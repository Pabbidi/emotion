from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import pandas as pd
import pickle   
import json

model  = pickle.load(open("models/random_forest_model.pkl", "rb"))
test_data = pd.read_csv("data/interim/test_bow.csv")
X_test = test_data.drop(columns=['label']).values
y_test = test_data['label'].values  

y_pred = model.predict(X_test)

metrics_dict = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_pred)
}

with open("reports/metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)
