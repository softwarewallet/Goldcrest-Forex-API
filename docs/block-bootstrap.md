# Dependence-Aware Block Bootstrap

`app.block_bootstrap.compare_with_block_bootstrap` compares aligned candidate and baseline net price changes using a moving-block percentile bootstrap. It resamples contiguous blocks of paired differences, rather than individual observations, to preserve some local serial dependence.

## Required controls

- Candidate and baseline observations must be chronologically aligned and use equivalent documented costs.
- The block length is mandatory and must be justified for the instrument, timeframe, and horizon. Report sensitivity across plausible block lengths.
- Record the random seed, resample count, confidence level, sample size, and interval with each result.
- This does not resolve nonstationarity, overlapping test windows, multiple testing, strategy selection bias, or dependence extending beyond the chosen blocks.
- An interval above zero is not proof of profitability and does not authorize model/strategy promotion.

This is offline research infrastructure only. GoldcrestAI remains `NO_TRADE` and disconnected from GoldcrestForex and all broker execution.
