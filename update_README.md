# LLM-Prompt-Recovery---Mistral7B-Meanprompt-Method

This project implements a mean prompt optimization solution based on Mistral 7B model for [LLM Prompt Recovery](https://www.kaggle.com/competitions/llm-prompt-recovery/overview). The `Mistral 7B Prompt Recovery` notebook is the **original baseline**, while `Mistral7B-Meanprompt-Method` is a **modified experimental variant** that introduces a different prompt aggregation and fallback strategy.

## 📋Project Overview
### Mistral 7B Prompt Recovery ([Original File](https://www.kaggle.com/code/richolson/mistral-7b-prompt-recovery-version-2))
The performance is evaluated using sentence-t5-base to compute embedding vectors for submission-ground truth pairs, followed by Sharpened Cosine Similarity (exponent=3) to score each pair, achieving **0.6215** on the public leaderboard and **0.6219** on the private leaderboard.

-   Direct prompt reconstruction per sample.
    
-   No global aggregation logic.
    
-   More diagnostic printing and timing utilities.

### Mistral7B-Meanprompt-Method (Updated Version)

This version is designed to be **engineering-friendly, extensible, and competition-ready**, yielding better results, with scores of **0.6483** on the public leaderboard and **0.6473** on the private leaderboard (The Kaggle competition has ended, and this is the score obtained after reproducing and submitting to Kaggle in my own environment).

-   Introduce a **mean / canonical prompt strategy**.
    
-   Add **rule-based fallback** for degenerate cases.
    
-   Simplify evaluation and remove some debug-only logic.
    
-   Aim for **more stable and leaderboard-safe predictions**.

## 🛠️Requirements
Recommended environment: **Python 3.9+**

Required packages:

-   torch
    
-   transformers
    
-   pandas
    
-   numpy
    
-   tqdm
    
-   accelerate

-   bitsandbytes
    
This project is designed to run in Kaggle Notebook environments. Most dependencies are preinstalled.
Only transformers, accelerate, and bitsandbytes may need to be added if they are not present.

## 📁Dataset
This project targets the **LLM Prompt Recovery**.
The dataset can be downloaded from the Kaggle competition [website](https://www.kaggle.com/competitions/llm-prompt-recovery/data). 

Required files (CSV or CSV.ZIP):

-   `sample_submission`
    
-   `test`
    
-   `train`

## 🚀 Usage
**1. Download the file**

Download the file and upload it to Kaggle.

```python
Mistral7B-Meanprompt-Method.ipynb
```

**2. Prepare Dataset**

Upload the competition dataset and [mistral-7b-it-v02 model](https://www.kaggle.com/datasets/ahmadsaladin/mistral-7b-it-v02) to `/kaggle/input/` (or adjust the related paths to point to your local dataset path).

**3. Run the Script and Get the Output**

The script generates a rewrite prompt words submission file (e.g., `submission.csv`).

![输入图片说明](/img/1.jpg)

## 🔧Key Design Changes

## 1. Mean / Canonical Prompt Introduction
### Mistral 7B Prompt Recovery (Before)

In prompt recovery tasks, some samples are effectively uninformative:
    
   -   `original_text == rewritten_text`
        
   -   Extremely short or generic rewrites.
   
   - Sometimes the model cannot reliably deduce prompts.

### Mistral7B-Meanprompt-Method (After)

A global baseline prompt is used as a fallback when the model cannot reliably infer a prompt.

```python
def find_dataset_dir():
base_line = "Please improve this text using the writing style... with maintaining the original meaning but altering the tone."
```

## 2. Rule-based Override for Degenerate Samples
### Mistral 7B Prompt Recovery (Before)

-   Always trust model output.
    
-   No post-processing correction.

### Mistral7B-Meanprompt-Method (After)

Explicitly patch problematic rows **after inference**.

```python
test_df.loc[
    test_df.original_text == test_df.rewritten_text,
    'rewrite_prompt'
] = base_line
```
**Advantages:**

-   Apply deterministic correction.
    
-   Reduce noise and extreme errors.

## 3. Reduced Debug / Timing Instrumentation
### Mistral 7B Prompt Recovery (Before)

Include explicit timing and verbose inspection:

```python
for row_index in rows_to_test:
    print(f"Actual Prompt: {row['rewrite_prompt']}")
    print(f"Predicted Prompt: {get_prompt(row['original_text'], row['rewritten_text'])}")
```

And timing estimates such as:

```python
print(f"Estimated {(elapsed_time_per_test * 1500) / 3600} hours for 1500 tests.")
```

### Mistral7B-Meanprompt-Method (After)

These sections are removed entirely.

```python
# # End timing
# # rows_to_test = [...]
```
**Advantages**

-   Prioritize operational stability and predictability.
        
-   Debug utilities are removed to avoid accidental overhead or output pollution.

## 4. Shift in Philosophy: Precision → Robustness

| Aspect |Mistral 7B Prompt Recovery  | Mistral7B-Meanprompt-Method |
|--|--|--|
| Prompt inference |Fully model-driven  | Model + rule-based |
| Edge-case handling | None | Explicit |
| Failure mode | Hallucinated prompts | Canonical fallback |
| Competition safety | Medium | High |

`Mistral7B-Meanprompt-Method` intentionally **sacrifices some per-sample expressiveness** in exchange for:

-   Lower variance
    
-   Fewer catastrophic failures
    
-   More stable public/private LB behavior

## 5. Code Structure Comparison

**Mistral 7B Prompt Recovery (Before)**

```python
Input texts
   ↓
Prompt reconstruction (LLM)
   ↓
Direct output (no correction)

```

**Mistral7B-Meanprompt-Method (After)**

```python
Input texts
   ↓
Prompt reconstruction (LLM)
   ↓
Rule-based validation
   ↓
Mean / baseline prompt fallback
   ↓
Final output

```

## 📝 Notice

- The feature engineering process requires a large amount of computation, and it is recommended to run it on a machine with at least 16GB of memory.