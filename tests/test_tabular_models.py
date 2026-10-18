import pytest

from app.models.tabular import LightGBMResearchModel, XGBoostResearchModel

FEATURES = ("momentum", "volatility", "trend")
ROWS = [{"momentum": float(i % 7) / 7, "volatility": float(i % 5) / 5, "trend": float((i * 3) % 11) / 11} for i in range(80)]
LABELS = [int(row["momentum"] + row["trend"] > 0.8) for row in ROWS]

@pytest.mark.parametrize(("model_class", "filename"), [(LightGBMResearchModel, "candidate.txt"), (XGBoostResearchModel, "candidate.json")])
def test_train_predict_save_load(model_class, filename, tmp_path):
    model = model_class(FEATURES, model_version="test-fixture-v1").fit(ROWS, LABELS, params={"n_estimators": 12})
    before = model.predict(ROWS[0])
    assert 0.0 <= before.probability_class_1 <= 1.0
    assert before.actionable is False
    artifact = tmp_path / filename
    model.save(artifact)
    after = model_class.load(artifact).predict(ROWS[0])
    assert after.model_name == before.model_name
    assert after.model_version == before.model_version
    assert after.probability_class_1 == pytest.approx(before.probability_class_1, abs=1e-6)

def test_feature_schema_and_non_finite_values_rejected():
    model = LightGBMResearchModel(FEATURES)
    with pytest.raises(ValueError, match="schema mismatch"):
        model.fit([{"momentum": 0.1, "volatility": 0.2, "other": 0.3}], [0])
    row = dict(ROWS[0], momentum=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        model.fit([row], [0])

def test_labels_must_be_explicit_binary_with_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        XGBoostResearchModel(FEATURES).fit(ROWS[:4], [1, 1, 1, 1])

def test_untrained_model_fails_closed():
    with pytest.raises(RuntimeError, match="not trained or loaded"):
        XGBoostResearchModel(FEATURES).predict(ROWS[0])
