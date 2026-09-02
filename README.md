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

## The paper

Synthetic tabular data is usually sold as a privacy solution: release the synthetic
table instead of the real one and the risk goes away. Foundation models have recently
been dropped into that role, especially in the low-data regime where classical
generators struggle and where practitioners in health, finance, and social science
actually work. I wanted to know whether that pitch holds up.

So I benchmarked foundation-model in-context learning generators — GPT-4o-mini,
LLaMA 3.3 70B, and TabPFN v2 — against deep generative baselines trained per dataset
(CTGAN, TVAE, TabDiff) and SMOTE. The foundation models are prompted with the dataset
schema, summary statistics, and seed rows and emit synthetic records with no
retraining. Each generator was measured on three axes: statistical fidelity via
marginal and joint distribution similarity, downstream utility via classifiers scored
with macro-average R² and ROC AUC, and privacy leakage via the *worst-case* result
across a suite of membership-inference attacks rather than a single attack or an
average.

Foundation models delivered on quality but not on privacy. Under low-data conditions
they produced the highest-fidelity, most useful tables in the study, and they leaked —
LLaMA 3.3 70B most of all, making it both the strongest generator on several quality
axes and the most vulnerable. That pairing is the core tension of the paper. Worst-case
evaluation turned out to matter: averaging hides the datasets and records where leakage
is severe, and which attack wins depends on the generator. Prompt-level adjustments
(smaller batch size, lower sampling temperature, summary statistics in the prompt)
measurably reduced leakage while preserving most of the fidelity gains — mitigation,
not elimination, but cheap to adopt since none of it requires retraining.

The practical takeaway: don't treat "we used a foundation model" as a privacy argument.
Audit your own generator, on your own data, at your own sample size, against a
worst-case multi-attack evaluation.

## Future work

Extending the attack suite, integrating differential privacy into the generation
pipelines, and automating prompt optimization so the mitigations above can be tuned per
dataset rather than chosen by hand.

## Project directory

**This repository holds the code, not the experiment.** The real dataset splits, the
generated synthetic tables, the raw attack outputs, result tables, and figures are all
absent — they're large, they're regenerable, and some of the source data isn't mine to
distribute. Several local analysis and plotting scripts are likewise omitted, since
they only operate on result trees that aren't shipped. Expect to point these scripts at
your own data before anything runs.

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
  LTM_ablation_dataset_selection.py  Picks the highest-risk datasets to ablate on
  LTM_dataset_stats.py               Summary statistics for the benchmark datasets

synth_mia_script_updates/    Membership-inference attack suite and its runners
```
