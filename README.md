# LLM Prompt Recovery - Mistral7B-Mean-Method

This project implements a mean prompt optimization solution based on Mistral 7B model for the [LLM Prompt Recovery](https://www.kaggle.com/competitions/llm-prompt-recovery/overview). The `Mistral 7B Prompt Recovery (Version 2)` notebook is the **original baseline**, while `Mistral7B-Mean-Method` is a **modified experimental variant** that introduces a different prompt aggregation and fallback strategy.


## 📋Project Overview

### Mistral 7B Prompt Recovery (Version 2) (Original File)
**public:0.6215, private:0.6219**
https://www.kaggle.com/code/richolson/mistral-7b-prompt-recovery-version-2
-   Direct prompt reconstruction per sample
    
-   No global aggregation logic
    
-   More diagnostic printing and timing utilities
    


### Mistral7B-Mean-Method (Modified Version)
**public:0.6556, private:0.6577**

-   Introduces a **mean / canonical prompt strategy**
    
-   Adds **rule-based fallback** for degenerate cases
    
-   Simplifies evaluation and removes some debug-only logic
    
-   Aims for **more stable and leaderboard-safe predictions**

## 🛠️Requirements

Recommended : **Python 3.9+**

Required packages:

-   torch
    
-   transformers
    
-   pandas
    
-   numpy
    
-   tqdm
    
-   accelerate

-   bitsandbytes
    
This project is designed to run in Kaggle Notebook environments. Most dependencies are preinstalled.
Only transformers, accelerate and bitsandbytes may need to be added if not present.

## 📁Dataset
This project targets the Kaggle competition **LLM Prompt Recovery**.

https://www.kaggle.com/competitions/llm-prompt-recovery/data

Required files (CSV or CSV.ZIP):

-   `sample_submission`
    
-   `test`
    
-   `train`
    

## 🚀 Usage


 **1. Download the file**
Download the file and upload it to kaggle
```python
Mistral7B-Mean-Method.ipynb
```

 **2. Prepare Dataset**

Upload the competition dataset and [mistral-7b-it-v02 model](https://www.kaggle.com/datasets/ahmadsaladin/mistral-7b-it-v02) to `/kaggle/input/` (or adjust the related paths to point to your local dataset path).


**3. Run the Script and Get the Output**

The script generates a rewrite prompt words submission file (e.g., `submission.csv`).
![输入图片说明](/img/1.jpg)



## 🔧Key Design Changes in Mistral7B-Mean-Method 


## 1. Mean / Canonical Prompt Introduction

### Mistral 7B Prompt Recovery (Version 2) (Before)
-   In prompt recovery tasks, some samples are effectively uninformative:
    
    -   `original_text == rewritten_text`
        
    -   extremely short or generic rewrites
   
    - Sometimes the model cannot reliably deduce prompts

### Mistral7B-Mean-Method (After)
A global baseline prompt used as a fallback when the model cannot reliably infer a prompt.

```python
base_line = "Please improve this text using the writing style... with maintaining the original meaning but altering the tone."
```


## 2. Rule-based Override for Degenerate Samples

### Mistral 7B Prompt Recovery (Version 2) (Before)

-   Always trusts model output
    
-   No post-processing correction


### Mistral7B-Mean-Method (After)


Explicitly patches problematic rows **after inference**

```python
test_df.loc[
    test_df.original_text == test_df.rewritten_text,
    'rewrite_prompt'
] = base_line

```
**Advantages**

-   Applies deterministic correction
    
-   Reduces noise and extreme errors

## 3. Reduced Debug / Timing Instrumentation

### Mistral 7B Prompt Recovery (Version 2) (Before)

Includes explicit timing and verbose inspection:

```python
for row_index in rows_to_test:
    print(f"Actual Prompt: {row['rewrite_prompt']}")
    print(f"Predicted Prompt: {get_prompt(row['original_text'], row['rewritten_text'])}")

```
and timing estimates such as:

```python
print(f"Estimated {(elapsed_time_per_test * 1500) / 3600} hours for 1500 tests.")

```

### Mistral7B-Mean-Method (After)

these sections are removed entirely

```python
# # End timing
# # rows_to_test = [...]
```
**Advantages**
-   Kaggle inference notebooks prioritize:
    
    -   Stability
        
    -   Runtime predictability
        
-   Debug utilities are removed to avoid accidental overhead or output pollution

## 4. Shift in Philosophy: Precision → Robustness

| Aspect |Mistral 7B Prompt Recovery  | Mistral7B-Mean-Method |
|--|--|--|
| Prompt inference |Fully model-driven  | Model + rule-based |
| Edge-case handling | None | Explicit |
| Failure mode | Hallucinated prompts | Canonical fallback |
| Competition safety | Medium | High |

`Mistral7B-Mean-Method` intentionally **sacrifices some per-sample expressiveness** in exchange for:

-   Lower variance
    
-   Fewer catastrophic failures
    
-   More stable public/private LB behavior

###  Code Structure Comparison

**Mistral 7B Prompt Recovery (Baseline)**

```python
Input texts
   ↓
Prompt reconstruction (LLM)
   ↓
Direct output (no correction)

```
**Mistral7B-Mean-Method (Modified)**

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

## 📝6 Notice
-   The feature engineering process requires a large amount of computation, and it is recommended to run it on a machine with at least 16GB of memory.

