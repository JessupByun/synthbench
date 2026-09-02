# Risk in Context: Benchmarking Privacy Leakage of Foundation Models in Synthetic Tabular Data Generation

**Jessup Byun · Xiaofeng Lin · Joshua Ward · Guang Cheng** — UCLA Trustworthy AI Lab

[![arXiv](https://img.shields.io/badge/arXiv-2507.17066-b31b1b.svg)](https://doi.org/10.48550/arXiv.2507.17066)

📄 **Paper:** https://doi.org/10.48550/arXiv.2507.17066

Accepted to the **KDD 2025 Workshop on Trustworthy Agentic and Generative AI Evaluation**,
where I gave an oral presentation in Toronto, Canada.

```bibtex
@misc{byun2025riskincontext,
  title         = {Risk In Context: Benchmarking Privacy Leakage of Foundation Models
                   in Synthetic Tabular Data Generation},
  author        = {Byun, Jessup and Lin, Xiaofeng and Ward, Joshua and Cheng, Guang},
  year          = {2025},
  eprint        = {2507.17066},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2507.17066}
}
```

---

## What I looked at

Synthetic tabular data is usually sold as a privacy solution: release the synthetic
table instead of the real one and the risk goes away. Foundation models have recently
been dropped into that role, especially in the **low-data regime** where classical
generators struggle and where practitioners in health, finance, and social science
actually work.

I wanted to know whether that pitch holds up. So I benchmarked foundation-model
in-context learning generators — GPT-4o-mini, LLaMA 3.3 70B, and TabPFN v2 — against
established deep generative baselines (CTGAN, TVAE, TabDiff) and SMOTE, and measured
fidelity, downstream utility, and privacy leakage side by side under small real-data
budgets.

## What I found

**Foundation models deliver on quality, but not on privacy.** Under low-data conditions
they produced the highest-fidelity, most useful synthetic tables in the study — and
they leaked. Getting good synthetic data and getting private synthetic data turned out
to be separate problems, and the first does not imply the second.

**LLaMA 3.3 70B leaked the most.** It was the strongest generator on several quality
axes and simultaneously the most vulnerable to membership-inference attacks. That
pairing is the core tension in the paper: the models most worth using in the low-data
regime are the ones I'd most hesitate to release output from.

**Worst-case matters more than average-case.** I evaluated privacy as the worst-case
result across a suite of membership-inference attacks rather than a single attack or a
mean. Averaging hides the datasets and records where leakage is severe, and no single
attack is the strongest one everywhere — which attack wins depends on the generator.

**Prompt-level knobs help, and they're cheap.** Reducing batch size, lowering sampling
temperature, and including summary statistics in the prompt all measurably reduced
leakage while preserving most of the fidelity gains. These are configuration changes,
not retraining — which makes them practical to adopt. They mitigate rather than
eliminate the risk.

**The takeaway I'd give a practitioner:** don't treat "we used a foundation model" as a
privacy argument. Audit your own generator, on your own data, at your own sample size,
against a worst-case multi-attack evaluation — and be especially wary of SMOTE as a
"safe" low-data baseline.

## How the benchmark works

**Generators.** Foundation models via in-context learning: GPT-4o-mini and LLaMA 3.3
70B are prompted with the dataset schema, summary statistics, and seed rows to emit
synthetic records with no retraining; TabPFN v2 samples features autoregressively.
Data-specific baselines trained per dataset: TabDiff (diffusion over mixed numerical
and categorical features), CTGAN and TVAE (GAN and VAE for tabular data), and SMOTE
(k-nearest-neighbor interpolation).

**Metrics.** Statistical fidelity via marginal and joint distribution similarity;
downstream utility via classifiers (logistic regression, decision tree, random forest,
XGBoost, CatBoost) scored with macro-average R² and ROC AUC; privacy leakage via
worst-case membership-inference attack ROC AUC.

## Repository layout

This repo holds the **code**, not the data. Everything below is what you need to
re-run the benchmark yourself.

```
LTM_generation_evaluation/   Generation and evaluation pipelines
  LTM_llama_deployment.py      LLaMA 3.3 70B generation via Groq
  LTM_tabpfn_deployment.py     TabPFN v2 generation
  LTM_alfred_evaluation.py     Fidelity / utility / diversity pipeline
  LTM_privacy_leakage.py       Worst-case-attacker privacy table
  LTM_ablation_*.py            Prompt, temperature, and batch-size ablations
  LTM_barplot_across_sizes.py, LTM_plot_privacy_tradeoffs.py    Figures
  validate_synthetic_data.py   Schema and row-count checks on generated tables

synth_mia_script_updates/    Membership-inference attack suite
  synth_mia/attackers/         Classifier, DCR, DCR-Diff, DOMIAS, DPI,
                               Density Estimator, Gen-LRA, Local Neighborhood, LOGAN, MC
  LTM_Synth-MIA.py             Attack runner (with reference data)
  LTM_Synth-MIA_no_reference.py  Attack runner (no reference data)
```

Three directories the scripts expect but that aren't distributed here:
`LTM_data/LTM_real_data/` (train/test splits per dataset and sample size — the real
datasets are public tabular benchmarks, cited in the paper), `LTM_data/LTM_synthetic_data/`
(generated tables), and `LTM_evaluation/` (raw attack outputs, result tables, figures).
All three are large and fully regenerable from the scripts above. Point the scripts at
your own data laid out the same way.

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
```

```bash
pip install pandas numpy scikit-learn torch matplotlib tabulate python-dotenv groq tabpfn-extensions
```

LLaMA generation runs through the Groq API — add your key to a `.env` at the repo root:

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

## Reproducing the experiments

Each script has a config block or `main()` at the bottom where you set the dataset and
generator. Run them from the repo root, in this order:

```bash
python LTM_generation_evaluation/LTM_llama_deployment.py
```

```bash
python synth_mia_script_updates/LTM_Synth-MIA.py
```

```bash
python LTM_generation_evaluation/LTM_alfred_evaluation.py
```

```bash
python LTM_generation_evaluation/LTM_privacy_leakage.py
```

Everything here is open source and free to use for reproducing the experiments,
evaluating synthetic data quality, or running your own privacy audit.

## Future work

Extending the attack suite, integrating differential privacy into the generation
pipelines, and automating prompt optimization so the mitigations above can be tuned
per dataset rather than chosen by hand.

## People and credits

- **Jessup Byun** — Undergraduate Researcher, UCLA
- **Xiaofeng Lin** — PhD, Statistics, UCLA (research mentor)
- **Joshua Ward** — UCLA Trustworthy AI Lab
- **Prof. Guang Cheng** — Faculty Advisor, UCLA Trustworthy AI Lab

The attack suite builds on [synth-mia](https://github.com/safe-data-sharing/synth-mia).
Prompt design for LLM generation is adapted from example B.5 of *Curated LLM: Synergy
of LLMs and Data Curation for Tabular Augmentation in Low-Data Regimes* (Seedat, Huynh,
et al., [arXiv:2312.12112](https://arxiv.org/abs/2312.12112)).
