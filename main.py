import pandas as pd
from pathlib import Path

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

visits = X_train.merge(y_train, on="Index")
y = visits["target"]
print("hello")

from sklearn.dummy import DummyRegressor
from skore import evaluate

feature_cols = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
X = visits[feature_cols]
y = visits["target"]

dummy = DummyRegressor(strategy="mean")
report = evaluate(dummy, X, y)
report.metrics.rmse()

from skore_cli.agent._skore_file import SkoreConfig
from skore import Project, login

cfg = SkoreConfig.load(Path(__file__).parent)
login(mode="hub")
project = Project(name="bobathon-esilv", mode="hub", workspace=cfg.workspace)
project.put("01_dummy", report)