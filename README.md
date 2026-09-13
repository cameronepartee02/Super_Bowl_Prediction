# Predicting Super Bowl Ticket Prices: What a Model Can (and Can't) Tell a Fan About When to Buy

**A secondary-market pricing model built for a StubHub/SeatGeek-style audience — and an honest account of what it turned out to actually predict.**

---

## The Question

Super Bowl tickets are famously expensive, and famously volatile — prices swing based on the matchup, the spread, and how close you buy to kickoff. Fans shopping the secondary market (StubHub, SeatGeek, TickPick, TicketIQ) don't have a clean way to know whether today's price is a good one or whether waiting will pay off.

This project set out to answer that question directly: **when should a fan buy their ticket to get the best price?**

The first version of this project was framed around the NFL as the seller. That framing didn't survive contact with reality — the NFL doesn't sell Super Bowl tickets to the public at market rates; the vast majority of the secondary market is exactly that, secondary. Correcting that assumption early reshaped the whole project around resale platforms instead of the league, and it's worth naming here rather than editing out: catching a flawed premise before building on top of it is part of the work.

---

## Getting the Data Right

Historical secondary-market ticket prices aren't a clean, standardized dataset anyone publishes — they had to be assembled from individual reported price points across sources and years. Early in that process, some of the price panels found online turned out to be synthetic or fabricated rather than actually observed. Those were identified and thrown out before any real observations were locked in.

Once fabricated sources were ruled out, the usable signal was smaller than it looked: **only the Yahoo Sports 4-site average, covering 2022–2026, was consistent enough across years to build on.** Earlier years (2015–2021) mixed incompatible price types — asking price, actual sale price, "get-in" price — that aren't comparable to each other, so they were set aside for the core model rather than forced together.

That left a genuinely small dataset: **21 price observations across 9 unique Super Bowls (Super Bowl LII through LX).** There have only ever been 59 Super Bowls, period — this isn't a sample-size problem that more data collection fixes. Every modeling decision downstream had to respect that ceiling.

Two rows had a blank `days_out_approx` value. Rather than guess based on the price pattern, the original source notes were checked — both were labeled "reported final average," meaning the price was the last one recorded at or near kickoff. Both were filled in as day-0 observations on that basis. A useful side finding fell out of that check: in both cases, the final/day-0 price was *higher* than the price recorded 13 days out for the same game — the opposite of the "buy early to save" pattern the project's premise assumes.

---

## The Leakage Catch

The engineered feature set originally included several columns describing how the game actually turned out: `home_team_won`, `home_score`, `away_score`, `went_to_ot`, `point_margin`, `blowout`, `underdog_won`, plus `Attendance`.

The problem: a fan deciding whether to buy a ticket 13 days before kickoff has no access to the final score. Any model trained on outcome data would be leaking the answer key, not learning a real pre-game pricing signal — no matter how predictive those columns looked in isolation. All eight columns were removed from the feature set entirely once this was caught. Two more price-shaped columns, `anchor_price` and `price_pct_of_anchor`, were excluded on the same logic — both are mathematically derived from the target itself.

This is the single moment in the project that most directly tests whether the model actually understands the problem it's solving, and it's flagged here for that reason rather than buried as routine cleanup.

**Final v1 feature set (10 columns):** `days_out_approx`, 4 platform dummies (SeatGeek, StubHub, TickPick, TicketIQ), home/away QB repeat-Super-Bowl-appearance flags, home/away team appearance numbers, and the Vegas `spread_line`.

---

## Why Leave-One-Group-Out Cross-Validation

With 9 unique games and 21 rows total, a single train/test split would be one noisy roll of the dice — whichever game or games happened to land in the holdout would swing the result. It also risks a subtler leak: rows from the same game landing in both train and test.

**Leave-One-Group-Out cross-validation (LOGO-CV), grouped by game, was used instead.** Each of the 9 games is held out exactly once; the model trains on the other 8 games' rows and is scored only on the held-out game. That produces 9 independent error estimates instead of one, and it's the standard approach for small, grouped data like this — there's no separate "final" test set left over after LOGO-CV, because the cross-validation *is* the evaluation.

---

## The Tuning Journey

The model is a `GradientBoostingRegressor`, tuned across four rounds, all evaluated with the same LOGO-CV loop:

| Version | n_estimators | max_depth | min_samples_leaf | Mean RMSE | Mean MAE |
|---|---|---|---|---|---|
| V1 (baseline) | 50 | 2 | 3 | $2,911.98 | $2,789.37 |
| V2 | 50 | 1 | 5 | $2,783.79 | $2,665.52 |
| V3 | 30 | 1 | 5 | $2,774.88 | $2,667.30 |
| V4 (final) | 15 | 1 | 5 | **$2,693.58** | **$2,604.83** |

The biggest jump came from V1 → V2 — shrinking `max_depth` and raising `min_samples_leaf`, both of which limit how complex any single tree can get. That mattered more than tree count: continuing to cut `n_estimators` (V2 → V3 → V4) kept helping modestly rather than plateauing.

Against a dataset average ticket price of **$8,452**, the final model's ~$2,605 MAE works out to roughly **31% relative error** — past "decent" and into "meaningful miss" territory by a standard rule of thumb (under ~10% is strong, 25–30%+ is a real miss). That's not a result to soften: it's the direct, explainable consequence of training on 9 total data points, a ceiling no amount of feature engineering can raise. Three specific games — Super Bowl LIII, LV, and LVIII — were consistently harder to predict across every tuning version, a real pattern rather than noise.

---

## The Finding That Mattered Most: Accuracy vs. the Premise

This is the part of the project that surprised me. Two versions of the final model were compared by feature importance:

**The more accurate model (final, tuned settings):**

| Feature | Importance |
|---|---|
| `home_qb_repeat_appearance` | 0.499 |
| `spread_line` | 0.284 |
| `away_qb_repeat_appearance` | 0.113 |
| `platform_StubHub` | 0.103 |
| `days_out_approx` | **0.000** |

The most accurate configuration of the model — the one with the lowest LOGO-CV error — assigns **zero importance to `days_out_approx`**, the exact feature the whole buy-timing premise depends on. A less-tuned version of the model (V1 settings) does use timing, at meaningful if modest importance, but at the cost of higher error.

Why: the tuned model uses shallow trees (`max_depth=1`), so each tree gets exactly one split. With that little room, the model concentrates entirely on its 3–4 strongest signals and never touches the rest. That's a direct, explainable side effect of tuning to fight overfitting on 21 rows — not a bug.

**The decision:** keep the more accurate model, and name the tension directly rather than quietly swapping in the less accurate one because it "uses the right feature." A portfolio piece that reports a clean final number is easy; naming a real tradeoff between what performs best and what matches the original premise is the harder and more honest version of the work.

---

## So What Does the Model Actually Do?

The original question was *"when should a fan buy, to get the best price?"* — a timing tool. Based on what the trained model actually uses, the honest answer is that it answers a different, related question: ***"roughly how pricey is this matchup going to be?"*** — a price-ballpark estimator driven by:

- Whether the home team's QB has been to a Super Bowl before (within the 2018–2026 window the data covers)
- Whether the away team's QB has
- The Vegas spread (how competitive the game is expected to be)
- Which platform the price is being quoted on

In practice, that means the inputs that matter are QB-been-before (yes/no) for both teams, the spread, and platform — not a date, and not full team or QB stats. It's a real, useful reframing, and one that only became visible by actually looking at feature importance rather than stopping at the RMSE.

---

## Limitations, Stated Plainly

- **The sample size has a hard ceiling.** One Super Bowl happens per year; this can't be fixed by "getting more data" the way most small-data problems can.
- **The most accurate model doesn't use the timing feature the project is framed around.** Left as an open, named tension rather than resolved by picking a worse-performing model.
- **Three games (LIII, LV, LVIII) were consistently hard to predict** across every tuning version — a real pattern, not something further tuning was likely to fix.
- **The QB repeat-appearance flag only sees the dataset's own window (2018–2026).** A QB with Super Bowl experience before that window will be under-flagged.
- **A simpler model type (Ridge regression) was identified as a worthwhile comparison and not yet run** — named here as future work rather than left out.

---

## Tools

Python (pandas, scikit-learn), Gradient Boosting Regression, Leave-One-Group-Out cross-validation.

*Full technical documentation, including the complete feature-selection rationale and tuning log, is available in the project repository.*

<img width="1411" height="671" alt="Screenshot 2026-09-10 at 8 04 10 PM" src="https://github.com/user-attachments/assets/05924659-651c-4cb2-8672-3680ed89e1bc" />

<img width="1422" height="671" alt="Screenshot 2026-09-10 at 8 04 22 PM" src="https://github.com/user-attachments/assets/af7cf2fb-d93a-4896-b866-af1fc22ef719" />
