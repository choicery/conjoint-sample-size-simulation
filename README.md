# Conjoint sample size simulation: data and analysis

Replication files for the simulation studies behind the article "Conjoint analysis sample size calculator" on
choicery.ai.

The calculator of the article plans two results of a conjoint study. This archive holds the simulations behind both,
and the scripts that rebuild the numbers of the article from them.

1. **The price premium of a feature.** How far can the premium that a market simulation reports for one level of an
   attribute be from the true premium? This is the whole error after an estimation with Hierarchical Bayes.
2. **The choice share.** The calculator shows its sampling error: how far the share among the respondents can be from
   the share in the target group, only because of who takes part. The estimation adds an error of its own. Its size
   is documented in `FINDINGS.md`.

License: CC BY 4.0 (Creative Commons Attribution 4.0 International) for all files, see `LICENSE.txt`. You may share and
adapt them if you give appropriate credit.

## How to cite

Bengart, P. (2026). *Conjoint sample size simulation: Data and analysis* (Version 1.0) [Data set]. Choicery.
https://github.com/choicery/conjoint-sample-size-simulation

    @misc{bengart2026simulation,
      author    = {Bengart, Paul},
      title     = {Conjoint sample size simulation: Data and analysis},
      year      = {2026},
      version   = {1.0},
      publisher = {Choicery},
      url       = {https://github.com/choicery/conjoint-sample-size-simulation}
    }

The same is in `CITATION.cff`, from which GitHub offers "Cite this repository".

## Quick start

    pip install numpy pandas scipy
    python analysis/reproduce.py --check
    python analysis/reproduce_more.py --check

`reproduce.py` rebuilds the formula for the whole error of a choice share and the coverage table (Table 1 in
`FINDINGS.md`). `reproduce_more.py` runs the scripts in `analysis/scripts/` on the price premium and on the prior
variance; the fit of the premium formula takes a few minutes. Both read the files in `data/`, print their results and compare them with the stored output. They need no
other software. Used versions: Python 3.14, numpy 2.3, pandas 2.3, scipy 1.18.

## The three formulas

### Price premium

    margin = K * g * sqrt(B^2 + V / n) * (levels / 2)^c      as a share of the tested price range
    B = 0.0032 * r^-1.08        the part of the error that more respondents do not reduce
    V = 0.136 * r^-1.06         the part that shrinks with the number of respondents n
    r = tasks * (options - 1)^0.19 / (sum of (levels - 1) over the attributes without the price + 1)
    K = 2.25                    for the margin that 9 out of 10 premiums stay within
    levels, c                   levels of the attribute that is priced; c = 0.43
    g                           1 when nearly everybody prefers the level; 1.12 when about 3 in 10 people prefer the
                                other level; 1.72 when almost half do, and then c = 0.14

The premium is the price increase at which a product is back at its old choice share after one of its levels was
improved, taken as the median over 300 random competitive situations (the robust premium of Choicery's market
simulator). The price is one linear effect, so it counts as one utility in `r`.

The formula was fitted on 3,544 studies of the reference case (`analysis/scripts/fit_formula.py`): 17 designs with 6 to
16 tasks of 2 to 4 options and 10 to 30 utilities, 50 to 2,000 respondents, priced attributes with 2 to 12 levels. More
options per task add little: a task with 3 options counts like 1.14 tasks with 2 options, one with 4 options like 1.23.
The measured 90% point is 2.04; `K` = 2.25 adds 10%. With `K`, 93% of the 10,632 premiums are inside the margin, and
about 90 to 96% by design, all sample sizes of a design together. With a design left out of the fit and of the safety
margin, 89% of its premiums were inside in the weakest case and 92 to 96% for the other 16 designs. Single cells of
design, sample size, levels and agreement scatter more: of 73 cells with at least 90 premiums, 9 were under 90%, the
lowest at 81%. All three premiums of a study were inside together in 86% of the studies. On 768 further studies with
50, 100 and 250 respondents for 10 designs, which were not part of the fit, 92% of the premiums were inside, 89 to 96%
by design (`test_new_sample_sizes.py`). `figures.py` gives the numbers behind Figures 1 to 3 of the article.

The formula holds for the reference case: a level worth about 8% of the tested price range, a typical reaction to
price and CBC tasks. With a weak reaction to price 64% of the premiums are inside the margin, for a level worth about
28% of the price range 74%; there the margin is too narrow (last part of `fit_formula.py`). With dual-response and
best-worst tasks on the same people the error was 0.86 to 0.93 times as large (`question_formats.py`).

The studies cross 8, 12 and 16 tasks with 2, 3 and 4 options and include lean designs with 6 or 8 tasks and 20 or 30
utilities, so that the weights of tasks, options and utilities can be told apart. Counting every option fully
(exponent 1 instead of 0.19) fits clearly worse: the design with the lowest share then has 74% of its premiums inside
the 90% point, against 86% with the fitted exponent (first part of `fit_formula.py`).

### Sampling error of a choice share

    margin = k * sqrt(h * P * (1 - P) / n)        P = choice share, n = respondents
    k = 1.65 for a 90% margin, 1.96 for a 95% margin
    h = 0.35 for very similar, 0.56 for typical and 0.72 for very different preferences in the target group

`h` is the variance of the personal choice probabilities relative to `P * (1 - P)`. With `h` = 1 this is the formula of
a poll in which every person makes one definite choice. This error contains no estimation: it compares the true shares
of the drawn respondents with the true shares of the population. `data/sampling_error/` holds the two result files:
`h` for five spreads of the preferences between people, and the check on 520 drawn samples with 29,500 shares.

### Whole error of a choice share

    margin = k * sqrt(b^2 + v / n)          in percentage points of a simulated choice share
    b = 4.3 * r^-0.57                       the part of the error that more respondents do not reduce
    v = 5000 * r^-0.81                      the part that shrinks with the number of respondents n
    k = 1.65 for a 90% margin, 2.0 for a 95% margin
    r = tasks * (options - 1) / sum(levels - 1)        comparisons per utility

`r` counts at most 16 tasks and 4 options per task. The formula is for choice shares from share of preference in
markets with three or four products, estimated with Hierarchical Bayes at the default prior variance of Choicery (4.0).
It was fitted to the runs in `data/runs_fit` and rounded. The other run folders test it. The fixed part `b` becomes
smaller with a smaller prior variance (`data/prior_variance_runs`, Finding 6 in `FINDINGS.md`).

## What one study on the price premium is

One file in `data/premium_studies` is one simulated study:

1. A design with a price is generated with the design generator of Choicery: attributes with their numbers of
   levels, five prices from 10 to 30, tasks per respondent, options per task.
2. A population is drawn from the generator (see below). One level of one attribute is the feature that is priced:
   its utility gain has a mean and a spread between people (`addon.gain_mean`, `addon.gain_sd`). The reaction to price
   differs between people as well.
3. `n` simulated respondents answer the tasks. Each picks the option with the highest utility plus a Gumbel error.
4. Individual utilities are estimated with the Hierarchical Bayes estimation of Choicery with the defaults of the app,
   the price as a linear attribute.
5. For three base products, the market simulator of Choicery reports the robust premium of the feature
   (`rows[].app_median`). The true premium (`rows[].true_median`) is computed for the same 300 competitive situations
   from the true utilities of 20,000 people of the population, without code of the app.

| Field | Meaning |
|---|---|
| `family` | the block of runs: `addon` to `addon9`, `ext1`, `ext2`, and `fmt1` for the other question formats |
| `question_format` | `cbc`, `dual_response`, `best_worst` or `best_worst_with_buy_question` |
| `cbc_partner` | for `fmt1`: the CBC study with the same people, tasks and prices |
| `levels`, `tasks`, `options` | the design without the price; `addon.attr` is the position of the priced attribute |
| `price.values`, `price.span_mean` | the tested prices and the mean utility span over them (4.0 typical, 2.0 weak) |
| `addon.gain_mean`, `addon.gain_sd` | mean and spread of the utility gain of the priced level |
| `n`, `pop`, `rep` | respondents, number of the population, number of the data set drawn from it |
| `hb_prior_variance`, `arm` | setting of the estimation and name of the condition; a few arms use longer chains or another prior variance |
| `rows[].app_median`, `rows[].true_median` | premium of the market simulator and true premium, in price units |

## What one run on choice shares is

One run file in the folders `runs_...` and `prior_variance_runs` is one simulated study:

1. A design (attributes with their numbers of levels, tasks per respondent, options per task) is generated with
   the design generator of Choicery (balanced overlap).
2. A population is drawn: a mean utility and a spread between people for every level.
3. `n` simulated respondents are drawn from the population and answer the tasks. Each picks the option with the
   highest utility plus a Gumbel error.
4. Individual utilities are estimated with the Hierarchical Bayes estimation of Choicery, with the defaults of
   the app (10,000 burn-in iterations, 10,000 saved iterations, thinning 5).
5. Markets are simulated twice: with the true utilities of 200,000 people drawn from the population (the true
   share, as mean logit choice probability) and with the estimated utilities of the `n` respondents (the estimated
   share, from the market simulator of Choicery).

The error of a share is the estimated share minus the true share.

| Field | Meaning |
|---|---|
| `levels`, `tasks`, `options` | the design |
| `n` | number of simulated respondents |
| `pop`, `rep` | number of the population and of the data set drawn from it |
| `pop_seed`, `data_seed`, `hb_seed` | random seeds of population, respondents and estimation |
| `shares` | the standard markets: five markets with three products and one with four |
| `shares_extra` | further markets with two and with five products |
| `shares[].rule` | `share_of_preference`, `first_choice` or `rfc` (randomized first choice) |
| `shares[].pop` | true share in the population |
| `shares[].est` | share simulated from the estimated utilities |
| `shares[].insample` | share simulated from the true utilities of the `n` respondents |
| `diagnostics` | convergence diagnostics of the estimation |

## The folders in `data/`

On GitHub every folder with study or run files is stored as one ZIP file with the same name
(`data/premium_studies.zip` for `data/premium_studies/`). The scripts read the ZIP files directly. Unzip them inside
`data/` if you want to look at single files.

| Folder in `data/` | Files | Content | Used for |
|---|---:|---|---|
| `premium_studies` | 5,272 | price premium: 4,888 CBC studies with 6 to 16 tasks of 2 to 4 options and 50 to 2,000 respondents (4,800 of them with the default settings of the estimation), and 384 studies with other question formats | the formula of the price premium and its tests |
| `prior_variance_runs` | 246 | choice shares with prior variance 4.0, 1.0 and 0.25 and with longer chains, estimated on identical answers | Finding 6 |
| `sampling_error` | 2 | result files of the sampling error: `h` by spread between people, and the check on drawn samples | the sampling error of the calculator |
| `runs_fit` | 1,608 | 67 designs with 4 to 8 attributes, 3 to 5 levels, 6 to 16 tasks, 2 to 4 options; 4 populations each; 150, 400 and 1,000 respondents; 2 data sets each | fitting the formula of the whole error |
| `runs_small_samples` | 1,608 | the same designs and populations with 50, 100 and 250 respondents | test |
| `runs_new_populations_and_designs` | 264 | the example design of the article with 12 new populations, and 3 designs outside the 67 with 8 populations each | test |
| `runs_real_study_populations` | 432 | populations taken from 18 real conjoint studies, see below | test |
| `runs_earlier_simulation` | 1,608 | an earlier version of the simulation with a generator whose people differ more from each other | test |
| `runs_range` | 960 | 20 designs at the edges of the range: three attributes, two-level attributes, six levels, more than eight attributes; 8 populations each | test |
| `runs_question_formats` | 1,134 | choice shares with dual-response CBC, best-worst CBC and best-worst CBC with a buy question, with the people, tasks and markets of CBC runs | Finding 3 |

## The generator of the populations

`data/generator.json` holds the three numbers of the generator used in all folders except the real studies and
the earlier simulation:

- `mean_sd` 0.85: the mean utility of every level is drawn from a normal distribution with this standard deviation.
- `sd_median` 0.95 and `sd_logspread` 0.4: the standard deviation between people of every level is
  `sd_median * exp(0.4 * z)` with a standard normal `z`.

Utilities of different levels are independent in this generator. The three numbers were chosen so that the
generator matches the medians of 20 real conjoint studies that we re-estimated with Hierarchical Bayes.

## Populations from real studies

For 18 real conjoint studies, the simulated people were drawn from a normal distribution with the mean utilities
and the full covariance matrix that Hierarchical Bayes had estimated for the study. These populations have
correlations between utilities, which the generator does not have. Every study was simulated with two designs
(its own tasks and options, and a contrast design), with 150, 400 and 1,000 respondents.

The estimated parameters of these studies are not part of this archive, because the data belong to other
researchers. The studies have neutral names here (`study_01` to `study_18`). So these 432 runs can be recalculated
from the run files, but not simulated again from this archive. `data/real_studies_summary.csv` describes every
study with a few numbers:

| Column | Meaning |
|---|---|
| `attributes`, `levels`, `utilities` | size of the design |
| `size_of_mean_utilities` | root mean square of the mean part-worths over all levels |
| `differences_between_people` | root mean square of the standard deviations between people over all levels |
| `median_sd_between_people` | median of these standard deviations |
| `spread_of_sd_log` | standard deviation of their logarithms |
| `mean_abs_correlation_between_attributes` | mean absolute correlation between part-worths of different attributes |

The simulations on the price premium and on the sampling error use the generator only.

## The 250 published studies

The article also describes 250 choice-based conjoint studies from the literature. They are not part of this
archive, which holds the simulations only. The list of the publications behind them can be downloaded with the
articles on choicery.ai.

## Findings on the whole error of a choice share

`FINDINGS.md` holds the findings on the whole error of a choice share in full, with their figures in `figures/`:
the limit that the design sets, the test of the formula on runs it had not seen (Table 1), question formats and
simulator methods, the direction of the error for large and small shares, how often the winner is right, the prior
variance, and what the simulation does not cover.

## Original scripts

`original_scripts/` holds the scripts that produced the runs and the numbers of the article, as they ran. They
are kept for transparency. They call the code of the Choicery app (design generator, Hierarchical Bayes
estimation, market simulator), which is not part of this archive, and they expect the folder layout of the
project. File names and some comments are in German. In the first runs on choice shares the seed of the design
generator had no effect, so a rerun draws other tasks and gives slightly different numbers; the run files are the
record. `analysis/reproduce.py` and the scripts in `analysis/scripts/` are the self-contained versions of the central
steps and give the numbers of the article.

## What this archive does not cover

- The Hierarchical Bayes estimation itself. The files hold its results.
- The parameters of the real studies (see above).
- The drawn samples behind the sampling error. `data/sampling_error` holds the result files; the two scripts in
  `original_scripts/2026-11-23_stichprobenrechner` draw them again.
- `analysis/reproduce.py` covers the fit of the formula of the whole error and the coverage table. Further numbers of
  `FINDINGS.md` (other simulation rules, number of products, the product with the largest share, question formats)
  come from the same run files through the original scripts.
