# Jane Street Real-Time Market Data Forecasting

> Predict the return on each trade opportunity using real-time anonymized market signals.

---

## Dataset

The training data has **~5.7 million rows**. Each row represents a single moment in time when Jane Street's system spotted a potential trade on a specific financial instrument.

```
date_id (which day) + time_id (which moment) + symbol_id (which asset)
= one row = one opportunity
```

### Key Identifiers

| Column | Description |
|--------|-------------|
| `date_id` | Integer representing a trading day — roughly 7 years of data. Not a real calendar date. |
| `time_id` | A moment within a trading day. After day 700, there are consistently **968 time_ids per day**. |
| `symbol_id` | The financial asset — could be a stock, bond, futures contract, or currency pair. Identity is anonymized. There are **~39 unique symbols**. |

### Features

**79 anonymized market signals** (`feature_00` to `feature_78`)

| Feature Type | What It Likely Represents | Count |
|--------------|--------------------------|-------|
| Categorical | Market session, instrument type, or trading venue | 3 (`feature_09`, `feature_10`, `feature_11`) |
| Continuous numerical | Price signals, volume, momentum, volatility measures | 76 (remaining) |

> Features have missing values (`NaN`).

---

## Targets

There are **9 responder columns** (`responder_0` through `responder_8`). All represent the return on the same trade, measured at different time horizons.

| Column | Horizon | Role |
|--------|---------|------|
| `responder_6` | ~20-day rolling avg | **PRIMARY TARGET** |
| `responder_7` | ~120-day rolling avg | Auxiliary target |
| `responder_8` | ~4-day rolling avg | Auxiliary target |
| `responder_0` to `responder_5` | Various shorter horizons | Auxiliary targets |

**Analogy:** Imagine you buy a house. The "return" could be measured as profit after 1 week, 1 month, 1 year, or 5 years — same trade, different measurement windows.

Values are continuous, roughly centered around **0**.
- Positive = profitable trade
- Negative = loss

---

## Weights

Each row has a `weight` column — Jane Street's internal confidence score for that trade opportunity. Think of it as *"how much would we bet on this one?"*

- **Higher weight** = more important row = bigger impact on your score
- `weight` and `|responder_6|` are **negatively correlated** — Jane Street bets less on trades with wild/risky returns and more on steady, predictable ones

---

## Evaluation Metric

**Sample-weighted zero-mean R²** — a modified version of the standard R².

$$R^2 = 1 - \frac{\sum w_i (y_i - \hat{y}_i)^2}{\sum w_i \cdot y_i^2}$$

| Symbol | Meaning |
|--------|---------|
| `yᵢ` | True value of `responder_6` for row *i* |
| `ŷᵢ` | Your predicted value for row *i* |
| `wᵢ` | Weight for row *i* |

**Key differences from standard R²:**
- Denominator uses **zero** instead of the mean of `y` — because financial returns are roughly mean-zero
- Predicting `0` for everything gives **R² = 0**, not negative
- High-weight rows matter more — nailing them is crucial

---

## Online Learning

Update the model's weights after each day using the true labels — **online learning**.
