# Pre-registration

**Title (working).** Representation diagnostics do not predict forecast skill at low signal-to-noise

**Author.** Sunny Alex

**Date of this document.** 9 September 2026

**Status.** This file is committed to the public repository **before** any model in the
study is trained. Every quantitative claim it makes about what the study will find is a
prediction, not a description. The commit hash and timestamp of this file are the
evidence for that ordering; if the git history shows this file added or materially
changed after `zoo/` contains results, the pre-registration is void and the study should
be read as exploratory.

---

## 1. The question

Across a population of models trained on the same data, does a representation-similarity
score predict out-of-sample forecast skill?

This is not the question "can a neural network learn a Bayesian posterior" — it can, and
that is established (Bishop & Bonilla 2023; Shai et al. 2024). It is not the question
"does probe accuracy entail that the model uses the probed property" — it does not, and
that is established (Hewitt & Liang 2019; Ravichander et al. 2021). The gap this study
addresses is that the **quantitative** relationship between representation score and
downstream skill has, as far as the authors could establish, never been estimated in any
domain. It is assumed universally and measured nowhere.

The regime of interest is low signal-to-noise, which is where financial forecasting lives
and where the probing literature has not gone.

## 2. Estimand

Within each stratum $c = (\text{process}, \text{target}, T, \text{SNR})$:

$$\rho_S(c) \;=\; \operatorname{Spearman}\Big(\; \rho^2_{\text{adj}}\big(h_t,\ \pi_t\big),\;\; R^2_{\text{oos}} \big/ R^2_{\text{ceiling}} \;\Big)$$

taken across the models in that stratum, where $h_t$ is the model's hidden state, $\pi_t$
the exact posterior of the generating process, and the skill measure is out-of-sample
$R^2$ expressed as a fraction of the computable ceiling for that stratum.

**Stratification is not optional and the reason is stated here in advance.** Training
length $T$ is a common cause of both variables: more data improves the representation and
independently improves the readout. Pooling across $T$ would therefore induce a positive
correlation that is not evidence of diagnostic value, and would bias the study toward the
hypothesis it is testing. The primary result is the within-stratum correlation. Pooled
figures may be reported as secondary and must be labelled as confounded.

SNR is varied through process parameters and is the scientific axis. $T$ is the
estimation axis. They are not to be combined into a single "difficulty" variable.

## 3. Generating processes

Both are fixed here and will not be re-tuned after results are seen.

**Process A — linear-Gaussian (control arm).**

$$x_t = a x_{t-1} + q\varepsilon_t, \qquad r_t = x_t + s\eta_t, \qquad \varepsilon,\eta \sim \mathcal N(0,1)$$

with $a = 0.97$, $s = 0.30$, and $q \in \{0.01,\ 0.02,\ 0.05\}$ giving the three SNR
levels. The Kalman filter is exactly optimal here, so the posterior is analytic and every
ceiling has a closed form. This arm exists so that disagreement between theory and harness
is attributable to the harness.

**Process B — three-state hidden Markov model.**

Annualised volatilities $(45\%, 18\%, 10\%)$ and drifts $(-30\%, +8\%, +12\%)$, daily
scaling, transition matrix

$$P = \begin{pmatrix} 0.95 & 0.04 & 0.01 \\ 0.02 & 0.95 & 0.03 \\ 0.005 & 0.025 & 0.97\end{pmatrix}$$

SNR levels are produced by scaling the drift vector by $\{0.5,\ 1.0,\ 2.0\}$ for the drift
target and by scaling the volatility separation for the variance target. The exact
posterior is the forward recursion.

## 4. Targets and ceilings

**Drift.** $y_{t+1} = r_{t+1}$, conditional mean $\mu^\top P^\top \pi_t$.

**Variance.** $y_t = \frac1k\sum_{j=1}^k r_{t+j}^2$ at $k \in \{1, 5\}$, conditional mean
$\frac1k\sum_j (\pi_t^\top P^j)\cdot(\mu^2+\sigma^2)$.

Every skill number is reported as a fraction of the ceiling
$R^2_{\text{ceiling}} = \operatorname{Var}(m)/\operatorname{Var}(y)$, computed per stratum
on the same folds as the models, never on the full sample.

Reference values already computed (`probe_lab_02.py`): HMM drift filter ceiling 0.001187,
HMM variance filter ceiling 0.1559 at $k=1$ and 0.3403 at $k=5$; linear-Gaussian drift
filter ceiling 0.028348 at $q=0.02$.

## 5. Score panel, and the status of the work that chose it

**Primary:** adjusted CCA, $\rho^2_{\text{adj}} = 1-(1-\rho^2)\frac{T-1}{T-p-1}$.
**Reported alongside:** raw CCA, for comparability with published work.
**Empirical check:** Hewitt–Liang probe selectivity.
**Excluded:** SVCCA and linear CKA.

This panel was selected in `probe_lab_01.py` and `probe_lab_03.py` on synthetic
constructions with known ground truth, before any model was trained. **That work is
exploratory method-selection, not hypothesis testing, and is labelled as such in the
paper.** Its findings — that raw CCA inflates as $p/(T-1)$, that linear CKA carries the
opposite bias and reaches Spearman 0.0000 against truth on a balanced grid, that SVCCA
discards signal not resident in leading principal components, and that the Hewitt–Liang
control score tracks $p/(T-1)$ to three decimals — are reported as method results and are
not among the predictions below.

The panel is now frozen. No measure will be added or removed after this commit.

## 6. The zoo

| axis | levels |
|---|---|
| width $p$ | 2, 4, 8, 16, 32, 64, 128 |
| depth | 1, 2 |
| dropout | 0, 0.2 |
| seed | 5 |
| $T$ | 1 000, 5 000, 20 000 |
| process | linear-Gaussian, HMM |
| target | drift, variance ($k=1$, $k=5$) |
| SNR | 3 levels |

Architecture: single-layer or two-layer Elman RNN, tanh activation, linear readout,
trained by hand-derived backpropagation through time in NumPy with gradients verified
against finite differences to at least $10^{-6}$ relative error before the zoo runs.
Optimiser, learning-rate schedule, early-stopping rule and shrinkage of the readout are
fixed in `zoo/config.py` at this commit.

Splits are expanding walk-forward. The ceiling is recomputed per fold. Hidden states used
for scoring are taken from held-out folds only.

**Stopping rule.** The grid above is the whole study. No models will be added after
results are inspected. If compute forces a reduction, the reduction will be a uniform
subsample of seeds, decided before results are seen, and recorded in `DEVIATIONS.md`.

## 7. Predictions

These are the confirmatory set. Each is falsifiable as stated.

**P1.** On the **drift** target, the within-stratum $\rho_S$ has $|\rho_S| < 0.2$ in the
majority of strata, with a bootstrap 95% interval containing zero.

**P2.** On the **variance** target, $\rho_S > 0.4$ in the majority of strata, with a
bootstrap 95% interval excluding zero.

**P3.** The distinction in P1/P2 is governed by the **ceiling**, not by the identity of the
target: strata with $R^2_{\text{ceiling}} < 10^{-2}$ behave as in P1 and strata above it as
in P2, and a regression of $\rho_S$ on $\log R^2_{\text{ceiling}}$ across all strata has a
positive slope with an interval excluding zero.

**P4.** On the linear-Gaussian **variance placebo** — a target with a measured ceiling of
0.000825, i.e. a calibrated zero — $|\rho_S| < 0.2$. This is the study's false-positive
control. If P4 fails, P2 cannot be interpreted.

**P5.** Raw CCA increases with width at fixed skill; adjusted CCA does not. Operationally:
regressing each score on $\log p$ with skill and stratum as controls gives a positive
coefficient for raw CCA with an interval excluding zero, and an interval containing zero
for adjusted CCA.

**P6.** Skill declines with width on the drift target at fixed $T$, and does not on the
variance target.

**Primary hypothesis.** P1 and P2 jointly — that is, the *contrast*
$\rho_S^{\text{variance}} - \rho_S^{\text{drift}}$ is positive with a bootstrap interval
excluding zero. A single correlation is not the claim; the SNR-dependence is.

**Multiple testing.** Six confirmatory predictions, Holm correction across the family at
$\alpha = 0.05$. Within-stratum correlations reported for description carry no inferential
claim and are not corrected. Confidence intervals throughout are stationary-bootstrap,
1 000 resamples, block length chosen by the automatic rule and reported.

**What would sink the central claim.** P4 failing, or P3 failing while P1 and P2 both
hold. The first would mean the harness manufactures correlations; the second would mean
the drift/variance contrast is about something other than signal-to-noise, and the
mechanism story would need rebuilding rather than patching.

**Relationship to prior exploratory work.** P5 and P6 re-test observations made in earlier
unregistered work by the same author (an RNN whose hidden state encoded the filtering
posterior while forecasting worse than a two-parameter baseline, with narrow networks
outperforming wide ones). Those observations motivated this study and are **not**
independent evidence for it. They are reported in the paper as motivation, with the
present study as the registered test.

## 8. Deviations

Any departure from this document — parameter change, axis change, analysis change — is
recorded in `DEVIATIONS.md` with the date, the reason, and whether it was made before or
after the relevant results were seen. Analyses added after seeing results are reported as
exploratory in the paper, without exception and regardless of how compelling they look.

## 9. Replication

`make all` regenerates every figure and table in the paper from the raw simulated data.
Random seeds are fixed and recorded. The environment is pinned in `requirements.txt`.
No result in the paper depends on a file not in this repository.
