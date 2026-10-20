"""Descriptive fold uncertainty envelopes and explicit regime stratification."""
from __future__ import annotations
from math import isfinite
from typing import Any, Mapping, Sequence
_METRICS = ("brier_score", "log_loss", "accuracy_at_fixed_half", "expected_calibration_error")
def _finite(test: Any, key: str) -> float | None:
    if not isinstance(test, Mapping): return None
    value = test.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)): return None
    return float(value)
def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values); pos = (len(ordered)-1)*q; lo=int(pos); hi=min(lo+1,len(ordered)-1)
    return ordered[lo]*(1-(pos-lo))+ordered[hi]*(pos-lo)
def build_regime_uncertainty_audit(fold_results: Mapping[str, Sequence[Mapping[str, Any]]], paired_comparisons: Sequence[Mapping[str, Any]], fold_regime_annotations: Mapping[str, Any] | None = None, *, minimum_folds_per_regime: int = 2) -> dict[str, Any]:
    if isinstance(minimum_folds_per_regime,bool) or not isinstance(minimum_folds_per_regime,int) or minimum_folds_per_regime<2: raise ValueError("minimum_folds_per_regime must be an integer of at least 2")
    baselines={r.get("baseline_name") for r in paired_comparisons if isinstance(r,Mapping) and isinstance(r.get("baseline_name"),str)}
    candidates=sorted(n for n in fold_results if n not in baselines)
    indexed={n:{r["fold_index"]:r for r in fold_results[n] if isinstance(r,Mapping) and isinstance(r.get("fold_index"),int) and not isinstance(r.get("fold_index"),bool) and r["fold_index"]>=0} for n in candidates}
    common=set.intersection(*(set(indexed[n]) for n in candidates)) if candidates else set()
    complete=[i for i in sorted(common) if all(_finite(indexed[n][i].get("test"),"brier_score") is not None and _finite(indexed[n][i].get("test"),"log_loss") is not None for n in candidates)]
    labels={}; source=method=None; status="NO_REGIME_ANNOTATIONS"
    if fold_regime_annotations is not None:
        if not isinstance(fold_regime_annotations,Mapping): raise ValueError("fold_regime_annotations must be a JSON object")
        source,method,raw=fold_regime_annotations.get("source"),fold_regime_annotations.get("method"),fold_regime_annotations.get("folds")
        if not isinstance(source,str) or not source.strip(): raise ValueError("regime annotation source must be explicitly declared")
        if not isinstance(method,str) or not method.strip(): raise ValueError("regime annotation method must be explicitly declared")
        if not isinstance(raw,Mapping): raise ValueError("regime annotation folds must be an object")
        for key,value in raw.items():
            if not isinstance(key,str) or not key.isdigit(): raise ValueError("regime annotation fold keys must be non-negative integer strings")
            if not isinstance(value,str) or not value.strip(): raise ValueError("regime labels must be non-empty strings")
            labels[int(key)]=value.strip()
        status="DECLARED_REGIME_ANNOTATIONS"
    envelopes=[]; strata=[]
    for name in candidates:
        folds=[]
        for i in complete:
            test=indexed[name][i].get("test")
            folds.append({"fold_index":i,"sample_count":test.get("sample_count") if isinstance(test,Mapping) else None,**{m:_finite(test,m) for m in _METRICS}})
        metrics={}
        for metric in _METRICS:
            vals=[r[metric] for r in folds if r[metric] is not None]
            if vals: metrics[metric]={"fold_count":len(vals),"minimum":min(vals),"q1":_quantile(vals,.25),"median":_quantile(vals,.5),"q3":_quantile(vals,.75),"maximum":max(vals),"range":max(vals)-min(vals)}
        envelopes.append({"candidate_name":name,"fold_count":len(folds),"folds":folds,"metrics":metrics,"actionable":False})
        by_regime={}
        for row in folds:
            label=labels.get(row["fold_index"])
            if label: by_regime.setdefault(label,[]).append(row)
        for label in sorted(by_regime):
            rows=by_regime[label]
            if len(rows)<minimum_folds_per_regime: continue
            summary={}
            for metric in _METRICS:
                vals=[r[metric] for r in rows if r[metric] is not None]
                if vals: summary[metric]={"fold_count":len(vals),"median":_quantile(vals,.5),"minimum":min(vals),"maximum":max(vals),"range":max(vals)-min(vals)}
            strata.append({"candidate_name":name,"regime":label,"fold_indices":[r["fold_index"] for r in rows],"fold_count":len(rows),"metrics":summary,"actionable":False})
    return {"protocol":"PER_FOLD_DISPERSION_AND_EXPLICIT_REGIME_STRATIFICATION","candidate_count":len(candidates),"common_complete_test_fold_indices":complete,"uncertainty_envelopes":envelopes,"regime_annotations":{"status":status,"source":source,"method":method,"folds":{str(k):labels[k] for k in sorted(labels)},"minimum_folds_per_regime":minimum_folds_per_regime,"stratified_results":strata},"confidence_intervals_produced":False,"model_selection_performed":False,"test_metrics_used_for_selection":False,"actionable":False,"interpretation":"Fold-level quartiles and ranges are descriptive, not confidence intervals or sampling-uncertainty estimates. Regime groups appear only for caller-supplied labels with declared source and method; no regimes are inferred. Two folds is not evidence of statistical power. Overlapping test windows may reuse observations. No pooling, significance testing, model selection, P&L or profitability claim."}
