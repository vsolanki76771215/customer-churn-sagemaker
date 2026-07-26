import json
import os
import tarfile
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

with tarfile.open("/opt/ml/processing/model/model.tar.gz") as tar:
    tar.extractall(path="/opt/ml/processing/model")

model_candidates = [
    os.path.join("/opt/ml/processing/model", name)
    for name in os.listdir("/opt/ml/processing/model")
    if name not in {"model.tar.gz"}
]
model_file = next((p for p in model_candidates if os.path.isfile(p)), None)
if not model_file:
    raise FileNotFoundError("The extracted XGBoost model file was not found.")

test_path = "/opt/ml/processing/test/test.csv"
test_df = pd.read_csv(test_path, header=None)
y_true = test_df.iloc[:, 0].astype(int).to_numpy()
X_test = test_df.iloc[:, 1:]

booster = xgb.Booster()
booster.load_model(model_file)
probability = booster.predict(xgb.DMatrix(X_test))
prediction = (probability >= 0.5).astype(int)

report = {
    "classification_metrics": {
        "auc": {"value": float(roc_auc_score(y_true, probability))},
        "accuracy": {"value": float(accuracy_score(y_true, prediction))},
        "precision": {"value": float(precision_score(y_true, prediction, zero_division=0))},
        "recall": {"value": float(recall_score(y_true, prediction, zero_division=0))},
        "f1": {"value": float(f1_score(y_true, prediction, zero_division=0))},
    }
}
os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)
with open("/opt/ml/processing/evaluation/evaluation.json", "w") as f:
    json.dump(report, f)
print(json.dumps(report, indent=2))
