````markdown
# NLP Story Generation Project

This project focuses on generating **children’s stories for ages 6–8** using **Mistral-7B-Instruct**, prompt-based baselines, and **LoRA / QLoRA fine-tuning**.

The broader goal is to build a multimodal pipeline:

**theme → story → scenes / frames → image prompts → illustrations**

At the current stage, the project covers:
- prompt-based story generation baselines
- dataset preprocessing and filtering
- LoRA fine-tuning on Mistral
- checkpoint evaluation and comparison

---

## Project Goal

Generate children’s stories that:

- use **simple vocabulary**
- maintain a **warm, child-friendly tone**
- include **2–3 main characters**
- are suitable for **6–7 visual scenes**
- end with a **clear moral**
- avoid **scary, violent, dark, or adult content**

---

## Current Progress

### Completed
- zero-shot story generation with Mistral
- few-shot prompting baseline
- dataset selection and preprocessing
- structured output formatting
- QLoRA fine-tuning on Mistral-7B-Instruct-v0.2
- checkpoint-based comparison between:
  - base zero-shot
  - base few-shot
  - fine-tuned LoRA checkpoint

### In Progress
- refining story quality
- improving consistency in character handling and moral generation

### Planned Next
- scene / frame extraction from generated stories
- frame-wise prompt generation
- character consistency pipeline
- illustration generation for storybook-style outputs

---

## Model and Training Setup

### Base Model
- `mistralai/Mistral-7B-Instruct-v0.2`

### Fine-Tuning Method
- **LoRA / QLoRA**
- 4-bit quantization
- PEFT-based adapter training

### Target Output Format

```text
Title: ...
Characters: ...
Story:
...
Moral: ...
````

This structure is used throughout prompting, preprocessing, and fine-tuning.

---

## Repository Contents

This repository includes:

* preprocessing notebooks
* prompt baseline notebooks
* fine-tuning notebooks
* comparison outputs
* proposal / project documentation
* LoRA checkpoint artifacts for inference

Large raw processed datasets and oversized training artifacts are intentionally excluded from Git tracking where necessary.

---

## Workflow

### 1. Prompt Baselines

We first evaluate Mistral using:

* **zero-shot prompting**
* **few-shot prompting**

This establishes a baseline before fine-tuning.

### 2. Dataset Preprocessing

The dataset is cleaned and filtered to better match the task:

* age suitability: 6–8
* simpler language
* moral-oriented structure
* safer story content
* scene-friendly narratives
* reduced noise in character extraction

### 3. Fine-Tuning

We fine-tune Mistral using LoRA adapters to improve:

* output format consistency
* child-friendly style
* moral inclusion
* better task alignment without relying on long few-shot prompts

### 4. Evaluation

The generated stories are compared across:

* base zero-shot
* base few-shot
* fine-tuned LoRA checkpoint

Evaluation focuses on:

* structure and formatting
* readability for children
* moral clarity
* story coherence
* suitability for later frame generation

---

## Example Themes

The current system has been tested on themes such as:

* kindness
* honesty
* sharing
* patience
* teamwork

---

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

The project currently uses the following stack:

* PyTorch
* Transformers
* Datasets
* PEFT
* TRL
* Accelerate
* BitsAndBytes
* Pandas
* Scikit-learn
* NLTK
* Jupyter / IPython kernel tools

---

## Notes on Checkpoints

The fine-tuning stage produces LoRA adapter checkpoints rather than full model weights.

To run inference later:

1. load the base Mistral model
2. attach the LoRA adapter checkpoint
3. generate structured stories for new themes

---

## Current Observations

From experiments so far:

* **few-shot prompting** provides a strong baseline
* **LoRA fine-tuning** improves task-specific structure and reduces dependency on examples
* the model has learned formatting and style reasonably well
* the main remaining weaknesses are:

  * occasional character inconsistency
  * weaker moral endings in some generations
  * incomplete scene-level structure for downstream image generation

These observations directly motivate the next step: **frame extraction from generated stories**

---

## Next Step

The next phase of the project is:

### Scene / Frame Generation

Convert each story into **6–7 structured frames**, where each frame contains:

* scene summary
* characters present
* setting
* main action
* emotional tone
* visual cues

This will allow the project to move from text-only generation toward **illustrated storybook generation**.

---


```
```
