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

**Code Examples**:
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

### Project 2: Nucleotide GPT - Genomic Language Model Training
**Project Description:** This project involved the development and training of "Nucleotide GPT," a decoder-only transformer model designed to understand the language of the genome and classify genomic sequences in downstream tasks. The model utilizes a standard transformer architecture but processes DNA at the single-nucleotide level, avoiding k-mer tokenization to preserve sequence integrity. Key architectural choices include Rotary Positional Embeddings (ROPE) for capturing relative positional information, RMS Normalization for training stability, and optimizations like Flash Attention for efficient computation on TPUs using JAX. The model was first pre-trained on a large corpus of reference genomes from multiple species (human, mouse, macaque, etc.) using a causal language modeling objective (predicting the next nucleotide). This pre-training phase allows the model to learn fundamental patterns and grammar of genomic sequences. 

**Results:** Nucleotide GPT demonstrated strong performance on the Genome Understanding Evaluation (GUE) benchmark, achieving results comparable to state-of-the-art models like DNABERT-2 and Nucleotide Transformer variants across tasks such as promoter detection, splice site prediction, and transcription factor binding site (TFBS) prediction.

**Key Learnings:**
* Gained experience in implementing, training, and fine-tuning large transformer models for biological sequence analysis using JAX on TPUs.
* Learned about different architectural choices for genomic foundation models, including tokenization strategies (single-nucleotide vs. k-mer), positional embeddings (ROPE vs. absolute/ALiBi), and attention mechanisms (Flash Attention).   
* Developed an understanding of transfer learning in genomics: pre-training on large, diverse datasets before fine-tuning for specific downstream tasks like predicting chromatin architecture.   
* Explored model interpretability techniques for transformers in a biological context, including analyzing sequence embeddings with UMAP  and interpreting attention patterns to identify sequence features driving predictions.

**Code Examples:**
* RMS normalization: Applied before attention and feed-forward layers for stability
```
def rms_norm(x: jax.Array, gamma: jax.Array) -> jax.Array:
    """Apply RMS normalization."""
    # Calculate RMS of input 'x', add epsilon for numerical stability
    rms = jnp.sqrt(jnp.mean(jnp.astype(x, jnp.float32)**2, axis=-1, keepdims=True) + 1e-6)
    # Normalize 'x' by its RMS and scale by learnable parameter 'gamma'
    # Cast back to bfloat16 for efficiency
    return jnp.astype(gamma * x / rms, jnp.bfloat16)
```
* Core attention calculation: Computes scaled dot-product attention with masking
```
def attention(
    q: jax.Array, # Query [B, H, T_q, D_k]
    k: jax.Array, # Key   [B, H, T_k, D_k]
    v: jax.Array, # Value [B, H, T_k, D_v]
    mask: jax.Array, # Attention mask [B, 1, T_q, T_k]
    cfg: Config,
    internals: Any, # For logging intermediate values
    layer_idx: int,
) -> jax.Array: # Output [B, H, T_q, D_v]
    """Compute scaled dot-product attention."""
    # Scale factor for query-key dot products
    scale = q.shape[-1] ** -0.5
    # Compute raw attention scores (logits)
    qk = jnp.einsum("bhtd,bhTd->bhtT", q, k) * scale # [B, H, T_q, T_k]

    # Apply mask (e.g., causal mask, padding mask)
    # Set masked positions to a large negative value (-1e30)
    qk = jnp.where(mask, qk, -1e30)

    # Compute attention weights using softmax
    # Use float32 for numerical stability during softmax
    attn_weights = jax.nn.softmax(qk.astype(jnp.float32), axis=-1) # [B, H, T_q, T_k]

    # Store attention weights for analysis
    internals['layers'][layer_idx]['attn_scores'] = attn_weights

    # Compute weighted sum of values
    # Result is cast back to bfloat16
    output = jnp.einsum("bhtT,bhTd->bhtd", attn_weights, v).astype(jnp.bfloat16) # [B, H, T_q, D_v]
    return output
```
* Forward pass (simplified):
```
def forward_layer(
    x: jax.Array, # Input sequence embeddings [B, T, D_model]
    segment_ids: jax.Array, # Segment IDs for masking [B, T]
    layer: Layer, # Layer weights and normalization parameters
    sin: jax.Array, # Sinusoidal embeddings for RoPE [B, T, D_k/2]
    cos: jax.Array, # Cosine embeddings for RoPE [B, T, D_k/2]
    idx: int, # Layer index
    cfg: Config,
    cache: KVCache | None = None, # Optional KV cache for inference
    internals: Any = None, # For logging
) -> tuple[jax.Array, jax.Array, jax.Array]: # Output, updated K, updated V

    # --- Attention Block ---
    # 1. Pre-Attention RMS Normalization
    with jax.named_scope("attn_pre_norm"):
        attn_in = rms_norm(x, layer.attn_in_gamma)

    # 2. Compute Query, Key, Value projections
    with jax.named_scope("qkv_matmul"):
        q = jnp.einsum("btd,dhq->bhtq", attn_in, layer.q) # [B, H_q, T, D_k]
        k = jnp.einsum("btd,dhk->bhtk", attn_in, layer.k) # [B, H_k, T, D_k]
        v = jnp.einsum("btd,dhv->bhtv", attn_in, layer.v) # [B, H_k, T, D_v]

    # 3. Apply Rotary Positional Embeddings (RoPE)
    with jax.named_scope("rope"):
        q = apply_rotary_embedding(q, sin, cos)
        k = apply_rotary_embedding(k, sin, cos)

    # (Handle KV Caching for inference - omitted for simplicity)

    # 4. Compute Attention Output
    with jax.named_scope("attention"):
        # Create attention mask based on segment IDs and causality
        mask = make_attention_mask(q.shape[2], k.shape[2], segment_ids, segment_ids, q_offset=0, causal=cfg.causal)
        # Use standard attention or optimized kernel (Flash Attention)
        if cfg.use_attn_kernel and cache is None:
             attn_out = attention_kernel(q, k, v, segment_ids, segment_ids, cfg)
        else:
             attn_out = attention(q, k, v, segment_ids, segment_ids, q_offset=0, cfg, internals, idx)


    # 5. Project attention output back to model dimension
    with jax.named_scope("projection"):
        attn_out_proj = jnp.einsum("bhtq,hqd->btd", attn_out, layer.proj) # [B, T, D_model]

    # 6. First Residual Connection + Post-Attention Norm
    with jax.named_scope("residual_attn"):
        attn_out_norm = rms_norm(attn_out_proj, layer.attn_out_gamma)
        x = x + attn_out_norm # Add residual

    # --- Feed-Forward Block ---
    # 7. Pre-FeedForward RMS Normalization
    with jax.named_scope("ffn_pre_norm"):
        ff_in = rms_norm(x, layer.ff_in_gamma)

    # 8. Feed-Forward Network (e.g., MLP with GELU activation)
    with jax.named_scope("ffw"):
        ff_hidden = jax.nn.gelu(jnp.einsum("btd,df->btf", ff_in, layer.w1)) # [B, T, D_ff]
        ff_out = jnp.einsum("btf,fd->btd", ff_hidden, layer.w2) # [B, T, D_model]

    # 9. Second Residual Connection + Post-FeedForward Norm
    with jax.named_scope("residual_ffn"):
        ff_out_norm = rms_norm(ff_out, layer.ff_out_gamma)
        x = x + ff_out_norm # Add residual

    # (Optionally store intermediate activations for analysis)
    if cfg.return_sae_intermediates and idx == cfg.num_layers // 2:
         internals[f'layer_{idx}_activations'] = x

    # Return output embeddings and updated K/V (used in caching)
    return x, k, v
```
### Project 3: Sparse Autoencoder for Interpreting Genome Language Model Activations
**Project Description:** While large language models like Nucleotide GPT can learn complex patterns in genomic data, understanding how they represent this information internally remains a challenge. This project focused on applying a Sparse Autoencoder (SAE) to the intermediate activations of the pre-trained Nucleotide GPT model to uncover interpretable features within its learned representations.

An SAE is an unsupervised neural network trained to reconstruct its input via a bottleneck layer that is typically larger (overcomplete) than the input dimension but constrained by a sparsity penalty (L1 regularization). This encourages the SAE to learn a sparse, distributed code where only a few "features" in the latent space are active for any given input. The goal was to train an SAE on the activations from the middle layer (layer 6 of 12) of Nucleotide GPT (2048 dimensions), mapping them to a larger, sparser latent space (8192 dimensions) before reconstructing the original activations. By analyzing which DNA sequences maximally activate specific latent features in the trained SAE, we aimed to identify biologically meaningful patterns learned by the underlying transformer model. The SAE was implemented and trained using JAX.

**Analysis and Results:** The SAE was trained on the residual stream activations from Nucleotide GPT's middle layer. After training, we identified the top DNA sequences that maximally activated each of the 8192 latent SAE features. These sequences were then analyzed using bioinformatics tools (MEME Tomtom for motifs, Dfam for repeats) to determine if the learned features corresponded to known biological elements.
* Repetitive Element Detection: Several distinct SAE features were found to consistently activate on sequences corresponding to specific families of repetitive elements, including components of LTR retrotransposons (THE1D-int, MLT2A) and different functional regions of LINE-1 elements (5' UTR, ORF2, 3' UTR). This indicates the SAE successfully decomposed the transformer's representation into features sensitive to these common genomic elements.
* Potential Motif Alignment: One feature showed alignment to a known motif for the transcription factor ZNF460. While intriguing, this could also reflect the model recognizing repetitive sequences often bound by KRAB-zinc finger proteins like ZNF460, highlighting the complexity of interpreting learned features.

**Key Learnings:**
* Gained practical experience implementing and training Sparse Autoencoders using JAX, including defining the architecture, loss function (MSE reconstruction + L1 sparsity), and optimization steps.
* Learned techniques for applying SAEs to interpret the internal activations of a pre-trained transformer model.
* Developed a workflow for analyzing learned SAE features by identifying maximally activating input sequences and correlating them with biological annotations using standard bioinformatics tools (motif databases, repeat databases).
* Understood the concept of using overcomplete, sparse representations to potentially disentangle complex features learned by deep learning models.

**Code Examples:**
* SAE Forward Pass and Loss: Defines the autoencoder structure (encoder, ReLU, decoder) and calculates reconstruction (MSE) and sparsity (L1) losses.
```
def fwd_sae(sae_weights, activations):
    """
    Forward pass for the Sparse Autoencoder.

    Args:
        sae_weights: Dictionary containing 'expand' (encoder) and 'contract' (decoder) weights.
        activations: Input activations from the transformer model [Batch*Time, Dim_model].

    Returns:
        Tuple: (Total loss, Dictionary of intermediate values and losses).
    """
    # Encode: Project activations to the higher-dimensional latent space
    # [B*T, D_model] @ [D_model, D_sae] -> [B*T, D_sae]
    latents_pre_relu = jnp.einsum('bd,df->bf', activations, sae_weights['expand'])

    # Apply ReLU activation - encourages sparsity
    latents = jax.nn.relu(latents_pre_relu) # [B*T, D_sae]

    # Decode: Reconstruct original activations from the sparse latent representation
    # [B*T, D_sae] @ [D_sae, D_model] -> [B*T, D_model]
    reconstructed = jnp.einsum('bf,fd->bd', latents, sae_weights['contract'])

    # --- Loss Calculation ---
    # 1. Reconstruction Loss (Mean Squared Error)
    reconstruction_loss = jnp.mean((reconstructed - activations)**2)

    # 2. L1 Sparsity Loss (encourages latent features to be zero)
    # Applied to the activated latent features
    l1_loss = l1_coeff * jnp.mean(jnp.abs(latents))

    # Total loss combines reconstruction and sparsity penalties
    total_loss = reconstruction_loss + l1_loss

    # Return loss and dictionary of useful metrics/internals
    return total_loss, {
        'latents': latents, # Sparse activations
        'reconstruction_loss': reconstruction_loss,
        'l1_loss': l1_loss,
        'nonzero_fraction': jnp.mean(latents > 0) # Fraction of non-zero latent features
    }

# Get function to compute gradients along with loss and internals
grad_sae = jax.value_and_grad(fwd_sae, has_aux=True)
```
