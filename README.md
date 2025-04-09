# Code and Analysis Portfolio

Hello, welcome to my code and analysis portfolio. I am Master of Science in Health Data Science student at the University of California, San Francisco, where I have developed a capstone project exploring the potential and limitations of genomic language models (gLMs). Prior to this, I was a medical student at the University of Melbourne, Australia. In Fall 2025, I will begin as a PhD in Bioengineering student at Stanford University. 

## My Interests
My primary research interest lies at the intersection of deep learning and genomics, with a focus on leveraging large language models (LLMs) to decode the complexities of gene regulation and non-coding DNA.

Understanding how genes are expressed and regulated is crucial for comprehending biological development, health, and the origins of disease. The sheer scale and complexity of genomic data, especially the intricate rules governing gene regulation across vast stretches of DNA, present significant hurdles for traditional analysis.

This is where I see the transformative potential of deep learning. Inspired by their success in natural language processing, LLMs offer powerful capabilities uniquely suited to biological sequence analysis. Their ability to learn complex patterns and long-range dependencies directly from sequence data aligns remarkably well with the biological reality of regulatory elements influencing genes from afar.

I am particularly interested in applying and developing LLM-based approaches for:

* Predicting the function of DNA sequences, especially within the non-coding regions that orchestrate gene regulation.
* Identifying regulatory elements like enhancers and promoters and understanding their sequence determinants.
* Modeling the impact of genetic variants on gene regulation and function, contributing to insights into disease mechanisms and personalized medicine.

Beyond applying these methods, I have a strong interest in mechanistic interpretability - a new field dedicated to reverse-engineering neural networks like transformers to understand how they perform computations. Rather than treating models as "black boxes", mechanistic interpretability aims to map their internal components, such as attention heads and circuits, to human-understandable algorithms and concepts. This pursuit is driven not only by scientific curiosity to understand the emergent reasoning processes within these complex systems but also by the critical need to ensure the safety, reliability, and alignment of powerful AI models. Understanding the internal mechanisms of the LLMs used in genomics is crucial for validating the biological insights they generate and building trust in their predictions.

## Skills Summary
### Programming Languages
* Deep Learning and LLMs:
  * Pretraining and finetuning LLMs, specifically genome language models (gLMs)
  * Training models on TPUs using high-performance libraries (JAX)
  * Developing and applying deep learning models for genomic analysis
* Mechanistic Interpretability:
  * Training and utilizing techniques like sparse autoencoders (SAEs) to identify and understand human-interpretable latent features within neural networks
* Programming and Data Science:
  * Proficient in Python (including core data science libraries like NumPy, Pandas, Scikit-learn) and R
  * Solid foundation in traditional machine learning algorithms (e.g., Logistic Regression, Linear Regression, SVMs)
* Bioinformatics and Genomics Tools:
  * Familiar with standard bioinformatics pipeles and tools for sequence analysis, alignment, quantification, peak calling, and data manipulation:
    * Quality Control: FastQC
    * Alignment: STAR, HISAT2
    * Sequence Manipulation/Analysis: Samtools, BedTools
    * Quantification: Kallisto
    * Peak Calling: MACS
    * Visualization/Downstream Analysis: DeepTools
* Statistical Software:
  * Experience with statistical analysis using Stata

## Work Samples
### Project 1: Mendelian Randomization Analysis of Total Cholesterol and Alzheimer's Disease
**Project Description:** This project investigated the potential causal relationship between genetically predicted total cholesterol (TC) levels and the risk of Alzheimer's disease (AD). Observational studies have shown associations, but establishing causality is challenging due to confounding factors. Mendelian Randomization (MR) uses genetic variants associated with an exposure (TC) as instrumental variables to estimate the causal effect on an outcome (AD), mitigating traditional confounding. This analysis employed a two-sample MR approach using summary statistics from large genome-wide association studies (GWAS) for TC  and AD. The analysis was conducted in R using the TwoSampleMR and RadialMR packages.

**Analysis and Results**: The analysis involved several key steps:
1. Data Preparation: Formatting exposure (TC) and outcome (AD) GWAS summary statistics, followed by clumping to select independent genetic instruments for TC.
2. Harmonization: Ensuring the effect alleles and their effects corresponded between the exposure and outcome datasets.
3. Initial MR Analysis: Applying multiple MR methods (Inverse Variance Weighted (IVW), MR-Egger, Weighted Median, Weighted Mode) to estimate the causal effect. Initial results suggested a significant causal effect with the IVW and MR-Egger methods.   
4. Sensitivity Analyses: Conducting heterogeneity tests (mr_heterogeneity) and assessing horizontal pleiotropy (mr_pleiotropy_test). These tests indicated significant heterogeneity and potential pleiotropy, suggesting that some genetic variants might be influencing AD through pathways other than TC. Radial MR was used to identify potential outlier SNPs driving these issues.   
5. Outlier Removal & Re-analysis: Eight outlier SNPs identified by Radial MR were removed. The MR analysis was repeated on the dataset excluding these outliers.   
6. Final Results: After removing outliers, none of the MR methods showed a statistically significant causal effect of TC on AD. While heterogeneity persisted, the test for horizontal pleiotropy was no longer significant, increasing confidence in the null finding.   

**Key Learnings**: 
* Applied the principles of Mendelian Randomization to investigate a complex causal question in human health.
* Gained proficiency in using the TwoSampleMR and RadialMR R packages for data preparation, analysis, and visualization.
* Learned the importance of comprehensive sensitivity analyses (heterogeneity, pleiotropy, outlier detection) in MR studies to assess the validity of the underlying assumptions.
* Understood how outlier variants can significantly influence MR results and the importance of methods to detect and account for them.
* Developed skills in interpreting results from various MR methods and sensitivity tests to draw robust conclusions.

**Code Samples**:
* Loading libraries and formatting data:
```
# Packages
library(tidyverse)
library(TwoSampleMR)
library(RadialMR)

# Formatting exposure data (Total Cholesterol)
exposure_data <- read.csv(file = TC_file_path, sep = "\t")
exposure_formatted <- format_data(exposure_data, type ="exposure", ...)

# Formatting outcome data (Alzheimer's Disease)
outcome_data <- read.csv(file = AD_file_path, sep = "\t")
outcome_formatted <- format_data(outcome_data, type ="outcome", ...)
```
* Harmonizing data and performing MR:
```
# Clump exposure data to get independent SNPs
exposure_clumped <- exposure_formatted %>%
  filter(pval.exposure < 1e-6) %>%
  clump_data()

# Extract outcome data for the selected SNPs
outcome_clumped <- filter(outcome_formatted, SNP %in% exposure_clumped$SNP)

# Harmonize exposure and outcome data
harmonized_data <- harmonise_data(exposure_dat = exposure_clumped,
                                 outcome_dat = outcome_clumped)

# Perform MR analysis
results <- mr(harmonized_data, method_list = c("mr_egger_regression",
                                              "mr_weighted_median",
                                              "mr_ivw_fe",
                                              "mr_weighted_mode"))
```
