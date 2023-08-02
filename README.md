<div align="center">
  <img src="./static/assets/th1-icon-round.svg" width="100px"/>
  <br />
  <br />

</div>

Welcome to **LLM Benchmarker Suite**!

There are many packages that assist in evaluation of large language models. We take the best practices available along with our own optimizations to create one-stop method to evaluate Large Language Models holistically.
## Introduction

LLM Benchmarker Suite is a one-stop platform for large model evaluation, aiming to provide a fair, open, and reproducible benchmark for large model evaluation. Its main features includes:

- Static Evaluations - Uses OpenCompass package to for coverage on most of popular benchmarking datasets and large language models.
- Evaluation Levels - Dataset independent evaluations similar to the one [here](https://lmsys.org/vicuna_eval/)

This concept allows for multiple levels of evaluating the effectiveness of a large langugae model.

## Installation

  To install the `llm_benchmarking_suite` package, use the following pip command:

  ```bash
  pip install llm_benchmarking_suite
  ```

## Get Started

### Pre-requisites
Run the following to command in a linux machine to check _CUDA toolkit_ and _cuDNN_ is correctly configured. This will not run in macOS or Windows.
```bash
nvidia-smi
nvcc --version
```
### Environment Setup
1. `git clone https://github.com/TheoremOne/llm-benchmarker-suite.git`
2. `cd llm-benchmarker-suite`
3. `python3 -m venv venv`
4. `source ./venv/bin/activate`
5. `git submodule init && git submodule update`
6. `pip install -e .`

### Run static evaluations
```bash
cd opencompass && python opencompass/run.py configs/eval_demo.py -w outputs/demo
```


### Important Suite Tools:
The Suite comprises of tools to help you carry out metrics analysis on large language models in a bunch of different ways to suit your use case.

#### OpenCompass:
This is a static evaluation package meant to test the capabilities of a model.
```python
python opencompass/run.py configs/eval_demo.py -w outputs/demo
```
  - **Comprehensive support for models and datasets**: Pre-support for 20+ HuggingFace and API models, a model evaluation scheme of 50+ datasets with about 300,000 questions, comprehensively evaluating the capabilities of the models in five dimensions.

  -  **Efficient distributed evaluation**: One line command to implement task division and distributed evaluation, completing the full evaluation of billion-scale models in just a few hours.

  - **Diversified evaluation paradigms**: Support for zero-shot, few-shot, and chain-of-thought evaluations, combined with standard or dialogue type prompt templates, to easily stimulate the maximum performance of various models.

  - **Modular design with high extensibility**: Want to add new models or datasets, customize an advanced task division strategy, or even support a new cluster management system? Everything about LLM Benchmarker Suite can be easily expanded!

  - **Experiment management and reporting mechanism**: Use config files to fully record each experiment, support real-time reporting of results.

 
## **Metrics**
The `metrics` package is a Python library that provides various evaluation metrics commonly used to assess the performance of large language models. It includes functions to calculate metrics such as F1 score, accuracy, and BLEU score.

  ### Installation

  The `metrics` package is not available on PyPI and can be used as a standalone package. To integrate it into your project, you can directly copy the individual metric files from the `metrics` directory or clone the entire repository.

  ### Usage

  To use the `metrics` package, follow these steps:

  Import the specific metric functions into your Python script or notebook:

  ```python
  from metrics.f1_score import calculate_f1_score
  from metrics.accuracy import calculate_accuracy
  from metrics.bleu_score import calculate_bleu_score
  from metrics.utils import count_true_positives_negatives
  ```

  Use the imported functions to evaluate your large language models. For example, if you have predictions and true labels for a binary classification task:

  ```python
  # Assume we have predictions and true labels as follows:
  predictions = [1, 0, 1, 1, 0]
  true_labels = [1, 0, 0, 1, 1]

  # Calculate true positives and true negatives for binary classification
  true_positives, true_negatives = count_true_positives_negatives(predictions, true_labels, positive_label=1)

  # Calculate F1 score and accuracy
  f1_score = calculate_f1_score(true_positives, false_positives, false_negatives)
  accuracy = calculate_accuracy(true_positives, true_negatives, len(predictions))

  print("F1 Score:", f1_score)
  print("Accuracy:", accuracy)
  ```

  Additionally, the calculate_bleu_score function can be used to compute the BLEU score for evaluating language generation tasks:
  ```python
  # Example usage for BLEU score calculation
  reference_sentences = ["This is a reference sentence.", "Another reference sentence."]
  predicted_sentence = "This is a predicted sentence."

  bleu_score = calculate_bleu_score(reference_sentences, predicted_sentence)
  print("BLEU Score:", bleu_score)

  ```
  ### Notes
  - The metrics package provides simple and easy-to-use functions for calculating evaluation metrics. However, keep in mind that these functions expect the necessary input arguments (e.g., true positives, true negatives) based on the specific task you are evaluating.

  - The calculate_f1_score and calculate_accuracy functions are designed for binary classification tasks. If you are working with multiclass classification, you may need to adapt or extend these functions accordingly.

  - For BLEU score calculation, the nltk library is used. Ensure that you have installed it before running the code:
  `pip install nltk`
  - This package aims to offer a basic set of evaluation metrics commonly used in NLP tasks. Depending on your specific use case, you may need to incorporate additional or specialized metrics for comprehensive evaluation.

  Feel free to explore and modify the metrics package to suit your evaluation needs. By using these evaluation metrics, you can better understand the performance and effectiveness of your large language models across various tasks and datasets.

## Evaluation Levels:

  The `llm_benchmarking_suite` package is a Python library that provides a simple and generic interface to work with various language models. It supports loading pre-trained models from Hugging Face, as well as integration with proprietary language models like Anthropic and GPT. This package allows you to generate completions using these language models based on given input text.

### Datasets
  This is a utility package that allows efficient loading of popular datasets for evaluation. They use HuggingFace loaders by default.
  ```python
  from dataset.hellaswaq import load_hellaswaq_dataset
  from dataset.race import load_race_dataset

  # Example usage:
  hellaswaq_data = load_hellaswaq_dataset()
  race_data = load_race_dataset()

  print("HellaSWAQ dataset:", hellaswaq_data)
  print("RACE dataset:", race_data)
  ```

### Usage

  Go to `eval_levels/example.py` for example usage of _Evaluation Levels_.

### Notes
    
- This package provides a generic interface to work with language models. Replace the dummy code for proprietary language models with the actual API calls once you have access to the real APIs.

- For Hugging Face models, the model_name parameter should be set to the desired model name. You can choose from models like "gpt2," "gpt2-medium," or "gpt2-large" for larger models.

- For proprietary language models like Anthropic or GPT, replace the api_key and model_url parameters with the actual API key and model URL provided by the respective vendors.

- The generate_completions method takes the input text and additional keyword arguments specific to the model. Check the respective class definitions for more details on the supported arguments.

- Ensure that you have internet access to download Hugging Face models and access the proprietary language model APIs.

## Leaderboard

We provide [LLM Benchmarker Suite Leaderbaord](https://llm-evals.formula-labs.com/
) for community to rank all public models and API models. If you would like to join the evaluation, please provide the model repository URL or a standard API interface to the email address `abhijoy.sarkar@theoremone.co`.


## Dataset Support

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Language</b>
      </td>
      <td>
        <b>Knowledge</b>
      </td>
      <td>
        <b>Reasoning</b>
      </td>
      <td>
        <b>Comprehensive Examination</b>
      </td>
      <td>
        <b>Understanding</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
<details open>
<summary><b>Word Definition</b></summary>

- WiC
- SummEdits

</details>

<details open>
<summary><b>Idiom Learning</b></summary>

- CHID

</details>

<details open>
<summary><b>Semantic Similarity</b></summary>

- AFQMC
- BUSTM

</details>

<details open>
<summary><b>Coreference Resolution</b></summary>

- CLUEWSC
- WSC
- WinoGrande

</details>

<details open>
<summary><b>Translation</b></summary>

- Flores

</details>
      </td>
      <td>
<details open>
<summary><b>Knowledge Question Answering</b></summary>

- BoolQ
- CommonSenseQA
- NaturalQuestion
- TrivialQA

</details>

<details open>
<summary><b>Multi-language Question Answering</b></summary>

- TyDi-QA

</details>
      </td>
      <td>
<details open>
<summary><b>Textual Entailment</b></summary>

- CMNLI
- OCNLI
- OCNLI_FC
- AX-b
- AX-g
- CB
- RTE

</details>

<details open>
<summary><b>Commonsense Reasoning</b></summary>

- StoryCloze
- StoryCloze-CN (coming soon)
- COPA
- ReCoRD
- HellaSwag
- PIQA
- SIQA

</details>

<details open>
<summary><b>Mathematical Reasoning</b></summary>

- MATH
- GSM8K

</details>

<details open>
<summary><b>Theorem Application</b></summary>

- TheoremQA

</details>

<details open>
<summary><b>Code</b></summary>

- HumanEval
- MBPP

</details>

<details open>
<summary><b>Comprehensive Reasoning</b></summary>

- BBH

</details>
      </td>
      <td>
<details open>
<summary><b>Junior High, High School, University, Professional Examinations</b></summary>

- GAOKAO-2023
- CEval
- AGIEval
- MMLU
- GAOKAO-Bench
- MMLU-CN (coming soon)
- ARC

</details>
      </td>
      <td>
<details open>
<summary><b>Reading Comprehension</b></summary>

- C3
- CMRC
- DRCD
- MultiRC
- RACE

</details>

<details open>
<summary><b>Content Summary</b></summary>

- CSL
- LCSTS
- XSum

</details>

<details open>
<summary><b>Content Analysis</b></summary>

- EPRSTMT
- LAMBADA
- TNEWS

</details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

## Model Support

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Open-source Models</b>
      </td>
      <td>
        <b>API Models</b>
      </td>
      <!-- <td>
        <b>Custom Models</b>
      </td> -->
    </tr>
    <tr valign="top">
      <td>

- InternLM
- LLaMA
- Vicuna
- Alpaca
- Baichuan
- WizardLM
- ChatGLM-6B
- ChatGLM2-6B
- MPT
- Falcon
- TigerBot
- MOSS
- ...

</td>
<td>

- OpenAI
- Claude (coming soon)
- PaLM (coming soon)
- ……

</td>

<!--
- GLM
- ...

</td> -->

</tr>
  </tbody>
</table>

## Acknowledgements

Some code in this project is cited and modified from [OpenICL](https://github.com/Shark-NLP/OpenICL).
