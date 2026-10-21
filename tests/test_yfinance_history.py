from __future__ import annotations

import pandas as pd
import pytest

from app.yfinance_history import normalize_yfinance_symbol


def test_fx_symbol_maps_to_yahoo_notation():
    assert normalize_yfinance_symbol("EUR/USD") == "EURUSD=X"
    assert normalize_yfinance_symbol(" gbpusd=x ") == "GBPUSD=X"


def test_empty_symbol_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_yfinance_symbol("   ")
