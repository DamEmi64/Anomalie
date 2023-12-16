from river import datasets
from river import evaluate
from river import drift
from river import metrics
from river import tree

dataset = datasets.Elec2().take(3000)

model = drift.DriftRetrainingClassifier(
    model=tree.HoeffdingTreeClassifier(),
    drift_detector=drift.binary.DDM()
)

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric)
