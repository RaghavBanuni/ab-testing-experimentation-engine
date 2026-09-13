# A/B Testing & Experimentation Engine

An experimentation engine built around a single premise: in most A/B testing setups,
the statistics are not the hard part — the *stopping rule* is. This repository
implements always-valid sequential inference, variance reduction, and the
invalidation gates that decide whether a readout may be believed at all, and then
measures its own methods on synthetic data where the truth is known.

## The problem this exists to solve

A fixed-horizon t-test controls the false-positive rate at α on one condition: it is
evaluated **exactly once**, at a sample size committed to in advance.

Nobody works that way. Experiment dashboards are refreshed daily and experiments are
stopped when they look significant. Every extra look is another chance to cross the
threshold by luck, so the realised false-positive rate is strictly greater than α and
grows with the number of looks. The test is not wrong; the way it is *used* is.

This repository does not ask you to take that on faith. `ab_engine.simulate` runs the
same simulated experiments through four analysis strategies and reports the rejection
rate each one actually achieves:

| strategy | what it does | valid under peeking? |
|---|---|---|
| `fixed_final` | one test, at the pre-committed sample size | yes — but you may not peek |
| `fixed_peeking` | the same test at every look, stop on first hit | **no** — this is the bug |
| `msprt` | mixture sequential probability ratio test | yes, at every stopping time |
| `alpha_spending` | O'Brien–Fleming style incremental spending | yes, for a fixed look schedule |

With `--true-effect 0` the reported rate *is* the false-positive rate, so the
comparison is direct:

```bash
python -m ab_engine.cli simulate --true-effect 0 --n-looks 5
```

You will see `fixed_peeking` reject well above the nominal 0.05 while `fixed_final`,
`msprt` and `alpha_spending` sit at or below it. Raise `--n-looks` and the
`fixed_peeking` rate climbs further; the other three do not. (Exact figures move with
`--seed`; the ordering does not, and two tests in
`tests/test_sequential.py` assert it.)

Then set a real effect to see what the guarantee costs and what it buys:

```bash
python -m ab_engine.cli simulate --true-effect 0.25 --n-looks 5
```

`msprt` gives up a little power against `fixed_final` at the same sample size, and in
exchange stops early — compare the `mean stop fraction` column.

## How the always-valid guarantee works

Treat the estimated difference in means as `delta_hat ~ Normal(delta, V)`. Instead of
testing against one point alternative, mix over alternatives with a `Normal(0, τ²)`
prior. The mixture likelihood ratio has a closed form:

```
Λ = sqrt(V / (V + τ²)) · exp( delta_hat² · τ² / (2·V·(V + τ²)) )
```

Under the null this is a non-negative martingale with expectation 1, so Ville's
inequality bounds the probability it **ever** exceeds `1/α`:

```
P( ∃n : Λ_n ≥ 1/α | H₀ ) ≤ α
```

That bound holds simultaneously at every sample size — which is precisely the licence
to stop whenever you want. This is the mSPRT of Robbins (1970), in the form developed
for online experiments by Johari, Koomen, Pekelis and Walsh (2017). Inverting the test
gives a **confidence sequence**: an interval you may inspect continuously, whose
probability of *ever* missing the truth is at most α. It is wider than a
fixed-horizon interval, and that width is the entire price.

`τ` is the analyst's guess at the scale of effects worth detecting. It trades
sensitivity across the range of possible effects — smaller `τ` detects small effects
sooner and large ones later. It does **not** affect validity, provided it is chosen
before looking at the data.

**Stated limitation:** the martingale argument assumes `V` is known. In practice `V`
is estimated from the sample, so the guarantee is asymptotic rather than exact, and
should not be relied on at very small samples or on heavy-tailed metrics.

The alpha-spending alternative uses the Lan–DeMets (1983) O'Brien–Fleming function
`α(t) = 2(1 − Φ(z_{1−α/2} / √t))`, converted to per-look levels by taking increments.
That conversion is an **approximation, and a conservative one**: exact
group-sequential boundaries account for the correlation between successive looks
(Armitage, McPherson and Rowe, 1969) and are slightly less strict. Spending the
increments can only spend *at most* the intended total, so the error rate stays at or
below α — correctness is preserved and a little power is given up.

## What else is in here

**Deterministic bucketing** (`ab_engine/bucketing.py`). Assignment is a pure function
of `(unit, experiment)` via SHA-256 — no stored state, no coordination between
servers, and a returning user always lands in the same variant. Two details that are
usually wrong: assignment is salted **per experiment**, so concurrent experiments do
not confound each other; and the exposure ramp uses a **separate salt** from the
variant split, so ramping to 10% still splits that 10% evenly and raising the ramp
admits a strict superset of the units already in. Both are tested.

**CUPED variance reduction** (`ab_engine/cuped.py`). Most variance in a per-user
metric is the user being who they already were. Regressing out a pre-period covariate
leaves `Var(Y_adj) = Var(Y)·(1 − ρ²)` with the expectation untouched, so a covariate
correlated at 0.7 removes about half the variance and roughly halves the traffic
needed (Deng, Xu, Kohavi and Walker, 2013). Two conditions the implementation
enforces in its interface: the covariate must be measured *before* assignment, and θ
must be fitted on the **pooled** sample — a per-arm θ absorbs part of the treatment
effect itself.

**Sample ratio mismatch** (`ab_engine/diagnostics.py`). A χ² gate on arm sizes, run
*before* any effect is reported, at α = 0.001. The strict level is deliberate: the
test has enormous power at scale, so 5% would fire on trivial imbalance, while a real
bucketing bug produces a p-value orders of magnitude below any threshold. An SRM is
not a negative result — it means the units being compared are not the units that were
assigned, which breaks the randomisation everything else rests on. The CLI refuses to
print an effect when the gate fails, and returns exit code 1.

**Multiplicity** (`ab_engine/multiplicity.py`). Twenty metrics at 5% give a ~64%
chance of at least one false positive. Benjamini–Hochberg is the default for
exploratory metrics because controlling the *proportion* of false discoveries is what
a triage process cares about; Holm–Bonferroni is there for launch gates where a single
false positive is unacceptable. BH is implemented as a true **step-up** procedure —
rejecting everything below the largest passing rank, not only those individually
passing, which is the most common implementation bug and has its own test.

**Segments** (`ab_engine/segments.py`). Slicing is corrected for multiplicity, and
checked for Simpson's paradox: when the effect is positive in every segment but
negative in aggregate, the aggregate is a size-weighted average whose weights moved.
The engine reports the reversal and names the likely cause rather than silently
picking one answer.

**Power** (`ab_engine/power.py`). Sizing before the fact, in both directions — sample
size for a target MDE, and MDE for the traffic you actually have. The second is the
more useful one: if the achievable MDE exceeds any plausible effect, the experiment
cannot succeed and should be redesigned rather than run.

## Install

```bash
git clone https://github.com/RaghavBanuni/ab-testing-experimentation-engine.git
cd ab-testing-experimentation-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer. Dependencies are numpy, scipy and pandas.

## Use

Size the experiment first:

```bash
# how much traffic to detect a 0.05 absolute lift?
python -m ab_engine.cli power --sigma 1.0 --mde 0.05

# the more honest question: what can 50,000 units actually detect?
python -m ab_engine.cli power --sigma 1.0 --total-n 50000

# binary metric, stated as a relative lift
python -m ab_engine.cli power --baseline-rate 0.10 --relative-lift 0.05
```

Inspect the assignment function, including its SRM check:

```bash
python -m ab_engine.cli bucket --n-units 100000 --exposure 0.1
```

Read out an experiment from a CSV with `variant` and `metric` columns (plus optional
`pre_metric` for CUPED and `segment` for slicing):

```bash
python -m ab_engine.make_dataset --out experiment.csv --true-effect 0.05
python -m ab_engine.cli analyse experiment.csv
```

The readout runs in decision order: invalidation gates, then CUPED, then the
fixed-horizon result, then the always-valid result, then corrected segments. The two
intervals are printed side by side so the cost of continuous monitoring is visible
rather than argued about.

Use as a library:

```python
from ab_engine import assign, msprt, srm_check, welch_t_test

# assignment: stable, salted per experiment
variant = assign("user-123", "checkout-v2", ["control", "treatment"]).variant

# gate before you interpret anything
if srm_check([n_treatment, n_control]).mismatch:
    raise RuntimeError("allocation is broken; do not read the effect")

# always-valid readout, safe to call on every dashboard refresh
fixed = welch_t_test(treatment_values, control_values)
result = msprt(fixed.estimate, fixed.standard_error ** 2, tau=0.1, alpha=0.05)
if result.reject_null:
    ship()
```

## Tests

```bash
pytest -q
```

The suite is behavioural rather than snapshot-based. It checks that ramping exposure
never reshuffles assignment, that concurrent experiments assign independently, that
CUPED preserves the mean exactly while cutting variance, that BH behaves as a step-up
procedure, that the SRM gate fires on a 5000/4500 split but not on a designed 90/10
ramp, that MDE and sample size are consistent inverses, that a Simpson reversal is
caught, and — most importantly — that peeking on a fixed-horizon test inflates the
false-positive rate while both sequential methods hold their level on the same data.

## Layout

```
ab_engine/
  bucketing.py     deterministic, salted, mutually independent assignment
  power.py         sample size and minimum detectable effect
  fixed_horizon.py Welch t-test and two-proportion z-test (the baseline)
  sequential.py    mSPRT, confidence sequences, alpha spending
  cuped.py         pre-period variance reduction
  diagnostics.py   sample ratio mismatch gate
  multiplicity.py  Benjamini-Hochberg and Holm-Bonferroni
  segments.py      heterogeneous effects and Simpson's paradox
  simulate.py      the harness that scores the methods themselves
  make_dataset.py  synthetic experiment generator
  cli.py           power / bucket / analyse / simulate
tests/             behavioural tests, including the peeking simulations
```

## References

- Robbins, H. (1970). *Statistical methods related to the law of the iterated logarithm.*
- Johari, R., Koomen, P., Pekelis, L., Walsh, D. (2017). *Peeking at A/B tests: why it matters and what to do about it.*
- Lan, K.K.G., DeMets, D.L. (1983). *Discrete sequential boundaries for clinical trials.*
- Armitage, P., McPherson, C.K., Rowe, B.C. (1969). *Repeated significance tests on accumulating data.*
- Deng, A., Xu, Y., Kohavi, R., Walker, T. (2013). *Improving the sensitivity of online controlled experiments by utilizing pre-experiment data.*
- Benjamini, Y., Hochberg, Y. (1995). *Controlling the false discovery rate.*
- Kohavi, R., Tang, D., Xu, Y. (2020). *Trustworthy Online Controlled Experiments.*

## Licence

MIT — see [LICENSE](LICENSE).
