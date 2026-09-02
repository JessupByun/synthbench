# Risk in Context: Benchmarking Privacy Leakage of Foundation Models in Synthetic Tabular Data Generation

**Jessup Byun · Xiaofeng Lin · Joshua Ward · Guang Cheng** — UCLA Trustworthy AI Lab

[![arXiv](https://img.shields.io/badge/arXiv-2507.17066-b31b1b.svg)](https://doi.org/10.48550/arXiv.2507.17066)

📄 **Paper:** https://doi.org/10.48550/arXiv.2507.17066

Accepted to the **KDD 2025 Workshop on Trustworthy Agentic and Generative AI Evaluation**
and presented as an oral presentation in Toronto, Canada.

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

## Overview and Methodology

Synthetic tabular data is frequently presented as a privacy-preserving alternative to
releasing real records. Foundation models have recently been adopted for this purpose,
particularly in the low-data regime, where conventional generators perform poorly and
where practitioners in health, finance, and the social sciences most often operate.
This work examines whether that privacy claim withstands systematic evaluation.

We benchmark three foundation-model in-context learning generators — GPT-4o-mini,
LLaMA 3.3 70B, and TabPFN v2 — against deep generative baselines trained per dataset
(CTGAN, TVAE, TabDiff) and against SMOTE. The foundation models are conditioned on the
dataset schema, summary statistics, and seed rows, and produce synthetic records
without retraining. Each generator is assessed along three dimensions: statistical
fidelity, measured through marginal and joint distribution similarity; downstream
utility, measured through classifiers scored with macro-average R² and ROC AUC; and
privacy leakage, quantified as the worst-case result across a suite of
membership-inference attacks rather than a single attack or an average across attacks.

Our results indicate that foundation models achieve strong fidelity and utility but
incur substantial privacy cost. In the low-data regime they produce the
highest-fidelity and most useful synthetic tables in the study while exhibiting
considerable leakage, with LLaMA 3.3 70B the most vulnerable of the generators
evaluated despite ranking among the strongest on several quality dimensions. This
coupling of quality and exposure constitutes the central tension of the paper. The
worst-case framing proves consequential: averaging across attacks obscures the datasets
and records for which leakage is severe, and the most effective attack varies by
generator. Prompt-level adjustments — reduced batch size, lower sampling temperature,
and the inclusion of summary statistics in the prompt — measurably reduce leakage while
preserving most of the fidelity gains. These adjustments mitigate rather than eliminate
the risk, but require no retraining and are therefore inexpensive to adopt.

We conclude that the use of a foundation model does not by itself constitute a privacy
guarantee. Generators should be audited on the deploying party's own data, at the
relevant sample size, under a worst-case multi-attack evaluation.

## Future Work

Planned extensions include broadening the attack suite, integrating differential
privacy into the generation pipelines, and automating prompt optimization so that the
mitigations described above can be tuned per dataset rather than selected manually.

## Project Directory

This repository contains the code, not the experimental artifacts. The real dataset
splits, generated synthetic tables, raw attack outputs, result tables, and figures are
not included: they are large, they are regenerable from the code provided, and portions
of the source data cannot be redistributed. Several local analysis and plotting scripts
are likewise omitted, as they operate exclusively on result trees that are not
distributed here. These scripts must be directed to a local data layout before use.

```
generation/                  Synthetic data generation
  LTM_llama_deployment.py            LLaMA 3.3 70B generation via Groq
  LTM_llama_deployment_2.py          LLaMA generation for the ablation runs
  LTM_tabpfn_deployment.py           TabPFN v2 generation
  validate_synthetic_data.py         Schema and row-count checks on generated tables

evaluation/                  Fidelity, utility, and privacy evaluation
  LTM_alfred_evaluation.py           Fidelity / utility / diversity pipeline
  LTM_alfred_evaluation_ablation.py  Same pipeline over the ablation runs
  LTM_privacy_leakage.py             Worst-case-attacker privacy table
  LTM_ablation_dataset_selection.py  Selects the highest-risk datasets for ablation
  LTM_dataset_stats.py               Summary statistics for the benchmark datasets

synth_mia_script_updates/    Membership-inference attack suite and its runners
```
