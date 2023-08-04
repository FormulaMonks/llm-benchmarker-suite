<div align="center">
  <img src="./static/assets/th1-icon-round.svg" width="100px"/>
  <br />
  <br />
</div>

Welcome to **LLM Benchmarker Suite**!

There are many packages that assist in evaluation of Large Language Models (LLMs). We take the best practices available along with our own optimizations to create one-stop method to evaluate LLMs holistically.

## Introduction

LLM Benchmarker Suite is a one-stop platform for large model evaluation, aiming to provide a fair, open, and reproducible benchmark for large model evaluation. Its main features includes:

- **Static Evaluations** - Uses OpenCompass package to for coverage on most of popular benchmarking datasets and large language models.
- **Evaluation Levels** - Dataset independent evaluations similar to the one [here](https://lmsys.org/vicuna_eval/).
- **LLM-as-a-judge** - Uses FastChat's LLM-as-a-judge to evaluate your models with MT-bench questions and prompts which is a set of challenging multi-turn open-ended questions for evaluating chat assistants.
- **OpenAI Evals** - Evals is a framework for evaluating LLMs (large language models) or systems built using LLMs as components. It also includes an open-source registry of challenging evals.

#### Running evals
- Learn how to run existing evals: [run-evals.md](docs/run-evals.md).
- Familiarize yourself with the existing eval templates: [eval-templates.md](docs/eval-templates.md).

This concept allows for multiple levels of evaluating the effectiveness of a large language models.

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
1. Clone the repository
```bash
git clone https://github.com/TheoremOne/llm-benchmarker-suite.git
cd llm-benchmarker-suite
```

2. Create and activate a virtual environment
```bash
python3 -m venv venv
source ./venv/bin/activate
```

3. Install Poetry if not already installed
(Ensure Poetry is added to your system's PATH)
Run the following command if you haven't installed Poetry yet:
`curl -sSL https://install.python-poetry.org | python3 -`

4. Install dependencies and submodules
```bash
poetry install
git submodule init && git submodule update
```

5. Install the main package and submodules in editable mode
```bash
pip install -e .
cd opencompass && pip install -e .
cd ../FastChat && pip install -e ".[eval]"
```


### Run static evaluations
1. `
cd opencompass && python opencompass/run.py configs/eval_demo.py -w outputs/demo
`
2. To view the locally created metrics dashboard start a server at `cd ../static && python3 -m http.server 8000` and go to http://localhost:8000/ in your browser. Alternatively you can also compare you're evaluations with ours at [LLM Benchmarker Suite Leaderbaord](https://llm-evals.formula-labs.com/
).



### Important Suite Tools:
The Suite comprises of tools to help you carry out metrics analysis on large language models in a bunch of different ways to suit your use case.

#### OpenCompass:
This is a static evaluation package meant to test the capabilities of a model.
<!-- [![PyPI](https://badge.fury.io/py/opencompass.svg)](https://pypi.org/project/opencompass/) -->
[Website](https://opencompass.org.cn/) •
[Documentation](https://opencompass.readthedocs.io/en/latest/) • 
[Installation](https://opencompass.readthedocs.io/en/latest/get_started.html#installation) •
[Reporting Issues](https://github.com/InternLM/opencompass/issues/new/choose)

```shell
python opencompass/run.py configs/eval_demo.py -w outputs/demo
```
- **Comprehensive support for models and datasets**: Pre-support for 20+ HuggingFace and API models, a model evaluation scheme of 50+ datasets with about 300,000 questions, comprehensively evaluating the capabilities of the models in five dimensions.

-  **Efficient distributed evaluation**: One line command to implement task division and distributed evaluation, completing the full evaluation of billion-scale models in just a few hours.

- **Diversified evaluation paradigms**: Support for zero-shot, few-shot, and chain-of-thought evaluations, combined with standard or dialogue type prompt templates, to easily stimulate the maximum performance of various models.

- **Modular design with high extensibility**: Want to add new models or datasets, customize an advanced task division strategy, or even support a new cluster management system? Everything about LLM Benchmarker Suite can be easily expanded!

- **Experiment management and reporting mechanism**: Use config files to fully record each experiment, support real-time reporting of results.

#### FastChat's LLM-as-a-Judge:
[Paper](https://arxiv.org/abs/2306.05685) • [Leaderboard](https://chat.lmsys.org/?leaderboard) • [MT-bench Human Annotation Dataset](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) • [Chatbot Arena Conversation Dataset](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)

In this package, you can use MT-bench questions and prompts to evaluate your models with LLM-as-a-judge.

MT-bench is a set of challenging multi-turn open-ended questions for evaluating chat assistants.

To automate the evaluation process, we prompt strong LLMs like GPT-4 to act as judges and assess the quality of the models' responses.

##### Evaluate a model on MT-bench

###### Step 1. Generate model answers to MT-bench questions

```shell
python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID]
```

Arguments

- `[MODEL-PATH]` is the path to the weights, which can be a local folder or a Hugging Face repo ID.
- `[MODEL-ID]` is a name you give to the model.

Example

```shell
python gen_model_answer.py \
  --model-path lmsys/vicuna-7b-v1.3 \
  --model-id vicuna-7b-v1.3
```

The answers will be saved to `data/mt_bench/model_answer/[MODEL-ID].jsonl`.

To make sure FastChat loads the correct prompt template, see the supported models and how to add a new model [here](../../docs/model_support.md#how-to-support-a-new-model).

You can also specify `--num-gpus-per-model` for model parallelism (needed for large 65B models) and `--num-gpus-total` to parallelize answer generation with multiple GPUs.

###### Step 2. Generate GPT-4 judgments

There are several options to use GPT-4 as a judge, such as pairwise winrate and single-answer grading.

In MT-bench, we recommend single-answer grading as the default mode.

This mode asks GPT-4 to grade and give a score to model's answer directly without pairwise comparison.

For each turn, GPT-4 will give a score on a scale of 10. We then compute the average score on all turns.

```shell
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]
```

Example

```shell
python gen_judgment.py --parallel 2 --model-list \
  vicuna-13b-v1.3 \
  alpaca-13b \
  llama-13b \
  claude-v1 \
  gpt-3.5-turbo \
  gpt-4
```

The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_single.jsonl`

###### Step 3. Show MT-bench scores

Show the scores for selected models

```shell
python show_result.py --model-list \
    vicuna-13b-v1.3 \
    alpaca-13b \
    llama-13b \
    claude-v1 \
    gpt-3.5-turbo \
    gpt-4
```

Show all scores

```shell
python show_result.py
```

For more information on usage details, refer to the following [docs](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md).

---
#### OpenAI Evals
Evals is a framework for evaluating LLMs (large language models) or systems built using LLMs as components. It also includes an open-source registry of challenging evals.

An “eval” refers to a specific evaluation task that is used to measure the performance of a language model in a particular area, such as question answering or sentiment analysis. These evals are typically standardized benchmarks that allow for the comparison of different language models. The Eval framework provides a standardized interface for running these evals and collecting the results.

At its core, an eval is a dataset and an eval class that is defined in a YAML file. An example of an eval is shown below:

```yaml
test-match:
id: test-match.s1.simple-v0
description: Example eval that checks sampled text matches the expected output.
disclaimer: This is an example disclaimer.
metrics: [accuracy]
test-match.s1.simple-v0:
class: evals.elsuite.basic.match:Match
args:
  samples_jsonl: test_match/samples.jsonl
```

We can run the above eval with a simple command:

```bash
oaieval gpt-3.5-turbo test-match
```
Here we’re using the oaieval CLI to run this eval. We’re specifying the name of the completion function (gpt-3.5-turbo) and the name of the eval (test-match)

With Evals, we aim to make it as simple as possible to build an eval while writing as little code as possible. An "eval" is a task used to evaluate the quality of a system's behavior. To get started, we recommend that you follow these steps:

To get set up with evals, follow the [setup instructions](https://github.com/openai/evals/blob/main/README.md#setup).

Refer to the following [Jupyter Notebooks](https://github.com/openai/evals/tree/1fe7d7adb739b727fd1319c073deda4e336c14f7/examples) for example of usages.

For more information on usage details, refer to the following [docs](https://github.com/openai/evals/blob/main/README.md).

#### Metrics
The `metrics` package is a Python library that provides various evaluation metrics commonly used to assess the performance of large language models. It includes functions to calculate metrics such as F1 score, accuracy, and BLEU score.

##### Installation

The `metrics` package is not available on PyPI and can be used as a standalone package. To integrate it into your project, you can directly copy the individual metric files from the `metrics` directory or clone the entire repository.

##### Usage

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
##### Notes
- The metrics package provides simple and easy-to-use functions for calculating evaluation metrics. However, keep in mind that these functions expect the necessary input arguments (e.g. true positives, true negatives) based on the specific task you are evaluating.
- The `calculate_f1_score` and `calculate_accuracy` functions are designed for binary classification tasks. If you are working with multiclass classification, you may need to adapt or extend these functions accordingly.
- For BLEU score calculation, the **nltk** library is used. Ensure that you have installed it before running the code `pip install nltk`.
- This package aims to offer a basic set of evaluation metrics commonly used in NLP tasks. Depending on your specific use case, you may need to incorporate additional or specialized metrics for comprehensive evaluation.

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
- For proprietary language models like Anthropic or GPT, replace the `api_key` and `model_url` parameters with the actual API key and model URL provided by the respective vendors.
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
