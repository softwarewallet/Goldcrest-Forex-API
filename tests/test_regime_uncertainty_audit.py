import pytest
from app.regime_uncertainty_audit import build_regime_uncertainty_audit
def fold(i,brier=.2,loss=.5): return {"fold_index":i,"validation":{"sample_count":10},"test":{"sample_count":10,"brier_score":brier,"log_loss":loss,"accuracy_at_fixed_half":.6,"expected_calibration_error":.1},"actionable":False}
def test_fold_dispersion_is_descriptive():
 data={"candidate":[fold(0,.1,.3),fold(1,.2,.5),fold(2,.3,.7)],"training_prevalence":[fold(0),fold(1),fold(2)]}; audit=build_regime_uncertainty_audit(data,[{"candidate_name":"candidate","baseline_name":"training_prevalence"}]); row=audit["uncertainty_envelopes"][0]; assert row["metrics"]["brier_score"]["median"]==pytest.approx(.2); assert audit["confidence_intervals_produced"] is False and audit["actionable"] is False
def test_declared_regimes_and_small_groups():
 data={"candidate":[fold(0),fold(1),fold(2),fold(3)]}; labels={"source":"protocol-v1","method":"predeclared regimes","folds":{"0":"LOW","1":"LOW","2":"HIGH","3":"HIGH"}}; result=build_regime_uncertainty_audit(data,[],labels); assert [r["regime"] for r in result["regime_annotations"]["stratified_results"]]==["HIGH","LOW"]; assert all(r["fold_count"]==2 for r in result["regime_annotations"]["stratified_results"]); small={"source":"x","method":"y","folds":{"0":"LOW","1":"HIGH","2":"HIGH"}}; assert build_regime_uncertainty_audit({"candidate":[fold(0),fold(1),fold(2)]},[],small)["regime_annotations"]["stratified_results"]==[]
def test_annotation_source_required():
 with pytest.raises(ValueError,match="source must be explicitly declared"): build_regime_uncertainty_audit({"candidate":[fold(0)]},[],{"method":"x","folds":{"0":"LOW"}})
def test_only_common_complete_folds():
 data={"a":[fold(0),fold(1),fold(2)],"b":[fold(0),fold(1,float("nan")),fold(2)]}; assert build_regime_uncertainty_audit(data,[])["common_complete_test_fold_indices"]==[0,2]
