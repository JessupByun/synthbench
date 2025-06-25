# tabular synthetic data research

## Research Overview

- **Undergraduate Researcher:** Jessup Byun, UCLA  
- **Research Mentor:** Xiaofeng Lin, PhD Statistics, UCLA  
- **Faculty Advisor:** Prof. Guang Cheng, UCLA Trustworthy AI Lab  

---

This repository contains the codebase for my research conducted at the UCLA Trustworthy AI Lab, exploring methods of generating synthetic tabular data with a specific focus on assessing the fidelity, utility, diversity, and privacy of generated datasets.

Project Description

The research systematically benchmarks foundation-model-based in-context learning (ICL) tabular generators—specifically GPT-4o-mini, LLaMA 3.3 70B, and TabPFN v2—against established deep generative models (GANs, VAEs, and diffusion models) including CTGAN, TVAE, TabDiff, and SMOTE. The primary aim is to evaluate these models under low-data conditions, which are particularly challenging for traditional synthetic data generation approaches.

Methodology

Synthetic Data Generators

Foundation models via In-Context Learning (ICL):

GPT-4o-mini: Utilizes structured prompts containing dataset schema, statistical summaries, and seed examples to generate synthetic rows without model retraining.

LLaMA 3.3 70B: Similar in approach to GPT-4o-mini but implemented using the open-source LLaMA model.

TabPFN v2: Employs autoregressive sampling for generating features sequentially based on previously sampled features.

Data-specific Generators (trained per dataset):

TabDiff: Uses diffusion processes tailored to numerical and categorical features.

CTGAN and TVAE: Utilize GAN and VAE frameworks respectively, tailored specifically for mixed-type tabular data.

SMOTE: A k-nearest neighbor interpolation method primarily designed for class imbalance issues.

Evaluation Metrics

The models are evaluated across three key dimensions:

Statistical Fidelity: Measured via marginal and joint distribution similarities (Kolmogorov–Smirnov distance, Chi-square divergence, Total Variation Distance).

Downstream Utility: Assessed through machine learning classifiers (logistic regression, decision tree, random forest, XGBoost, CatBoost) evaluated with macro-average ROC AUC scores.

Privacy Leakage: Quantified by worst-case membership-inference attacks (MIAs), measuring leakage via ROC AUC performance.

Key Findings

Foundation models achieve high fidelity and utility, particularly beneficial under low-data scenarios, yet pose significant privacy risks.

LLaMA 3.3 70B demonstrated the highest privacy leakage compared to other models.

Simple prompt-level adjustments—such as reducing batch size, lowering sampling temperature, and including summary statistics—effectively mitigate privacy risks while maintaining a substantial portion of data fidelity.

Contributions

Comprehensive benchmarking of privacy leakage across synthetic data generators.

Identification of effective strategies (prompt tuning) for privacy mitigation in foundation-model-based synthetic data generation.

Usage

The repository includes scripts and detailed instructions for reproducing experiments, evaluating synthetic data quality, and performing privacy leakage audits.

Future Work

Planned extensions include exploring additional attack models, integrating differential privacy techniques, and automated prompt optimization to further improve the safety and efficacy of tabular synthetic data generation.
