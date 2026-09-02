# tabular synthetic data research

## Research Overview

This repository contains the codebase for my research conducted at the UCLA Trustworthy AI Lab, exploring synthetic tabular data generators with a specific focus on assessing the fidelity, utility, diversity, and privacy of synthetic data.

---

## Project Description

The research systematically benchmarks foundation-model-based in-context learning (ICL) tabular generators—specifically GPT-4o-mini, LLaMA 3.3 70B, and TabPFN v2—against established deep generative models (GANs, VAEs, and diffusion models), including CTGAN, TVAE, TabDiff, and SMOTE. The primary aim is to evaluate these models under low-data conditions, which are particularly challenging for traditional synthetic data generation approaches.

## Methodology

### Synthetic Data Generators

#### Foundation models via In-Context Learning (ICL):

**GPT-4o-mini:** Utilizes structured prompts containing dataset schema, statistical summaries, and seed examples to generate synthetic rows without model retraining.

**LLaMA 3.3 70B:** Similar in approach to GPT-4o-mini but implemented using the open-source LLaMA model.

**TabPFN v2:** Employs autoregressive sampling for generating features sequentially based on previously sampled features.

#### Data-specific Generators (trained per dataset):

**TabDiff:** Uses diffusion processes tailored to numerical and categorical features.

**CTGAN and TVAE:** Utilize GAN and VAE frameworks, respectively, tailored specifically for mixed-type tabular data.

**SMOTE:** A k-nearest neighbor interpolation method primarily designed for class imbalance issues.

## Evaluation Metrics

The models are evaluated across three key dimensions:

**Statistical Fidelity:** Measured via marginal and joint distribution similarities.

**Downstream Utility:** Assessed through machine learning classifiers (logistic regression, decision tree, random forest, XGBoost, CatBoost) evaluated with macro-average R^2 and ROC AUC scores.

**Privacy Leakage:** Quantified by worst-case membership-inference attacks (MIAs), measuring leakage via ROC AUC performance.

## Key Findings

Foundation models achieve high fidelity and utility, particularly beneficial under low-data scenarios, yet pose significant privacy risks.

LLaMA 3.3 70B demonstrated the highest privacy leakage compared to other models.

Simple prompt-level adjustments—such as reducing batch size, lowering sampling temperature, and including summary statistics—effectively mitigate privacy risks while maintaining a substantial portion of data fidelity.

## Contributions and Publication

This work culminated in a comprehensive benchmarking of privacy leakage across synthetic data generators and was accepted to the *KDD 2025 Workshop on Trustworthy Agentic and Generative AI Evaluation* with an an oral presentation session which was delivered in Toronto, Canada. Access the paper here: https://doi.org/10.48550/arXiv.2507.17066

## Usage

The repository includes open-source free-to-use scripts for reproducing experiments, evaluating synthetic data quality, and performing privacy leakage audits.

## Future Work

Planned extensions include exploring additional attack models, integrating differential privacy techniques, and automated prompt optimization to further improve the safety and efficacy of tabular synthetic data generation.
