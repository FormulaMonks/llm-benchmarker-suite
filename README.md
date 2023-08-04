<div align="center">
  <img src="./static/assets/th1-icon-round.svg" width="100px"/>
  <br />
  <br />

</div>

Welcome to **LLM Benchmarker Suite**!
Many packages assist in the evaluation of large language models. We take the best practices available along with our optimizations to create a one-stop method to evaluate Large Language Models holistically.

To use learn how to use the tools directly jump to [Header](#tools-overview) or [Table of Contents](#table-of-contents)

# <a id="table-of-contents"></a> Table of Contents
- [Introduction](#introduction)
- [Building Blocks of LLM Evaluation](#building-blocks-of-llm-evaluation)
- [Tools Overview](#tools-overview)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## <a id="introduction"></a> Introduction

In recent years, the field of natural language processing (NLP) has witnessed an unprecedented surge in innovation, largely fueled by the advancements in large language models (LLMs) such as GPT, BERT, and their derivatives. These models exhibit an impressive ability to understand and generate human-like text, revolutionizing various applications like text generation, sentiment analysis, language translation, and more. As their potential continues to unfold, researchers and practitioners alike are increasingly invested in benchmarking the performance of these LLMs to better comprehend their capabilities and limitations.

However, despite the widespread interest in benchmarking LLMs, the process itself has remained remarkably non-standardized and often lacks a coherent framework. This lack of standardization introduces significant challenges, making it difficult to compare results across studies, impeding the reproducibility of findings, and hindering the overall progress of the field. Consequently, there arises a pressing need to establish a comprehensive and standardized approach to benchmarking LLMs, ensuring that evaluations are meaningful, consistent, and comparable.

Motivated by this imperative, we introduce the **LLM Benchmarking Suite**, a pioneering effort aimed at addressing the current fragmentation and ambiguity surrounding LLM benchmarking. Our suite offers a structured methodology, a curated collection of diverse benchmarking tasks, and an amalgam of open-source toolkits designed to streamline the process of assessing LLM performance. By providing a unified platform for benchmarking, our project seeks to foster collaboration, promote transparency, and ultimately elevate the quality of research within the NLP community.

## <a id="building-blocks-of-llm-evaluation"></a> Building Blocks of LLM Evaluation
For manually conducting evaluations, you generally follow the following steps:
- Load the model
- Load an appropriate benchmarking dataset
- Selecting a relevant metric for evaluation
### <a id="loading-the-model"></a> Loading the Model
We need to figure out a way to load these models locally in the most efficient manner to run fast inefernces with. The following options support local loading of a model which differs based on operating system under usage:

- Ollama (Mac) [jmorganca/ollama: Get up and running with large language models locally (github.com)](https://github.com/jmorganca/ollama)
- GPT4ALL [(GPT4All)](https://gpt4all.io/index.html)
- LocalAI [(LocalAI :: LocalAI documentation)](https://localai.io/)
- oobabooga [(oobabooga/text-generation-webui: A gradio web UI for running Large - - Language Models like LLaMA, llama.cpp, GPT-J, OPT, and GALACTICA. (github.com)](https://github.com/oobabooga/text-generation-webui)
- LocalGPT [(PromtEngineer/localGPT: Chat with your documents on your local device using GPT models. No data leaves your device and 100% private. (github.com))](https://github.com/PromtEngineer/localGPT)
- Llama2-webui [(liltom-eth/llama2-webui: Run Llama 2 locally with gradio UI on GPU or CPU from anywhere (Linux/Windows/Mac). Supporting Llama-2-7B/13B/70B with 8-bit, 4-bit. Supporting GPU inference (6 GB VRAM) and CPU inference. (github.com)](https://github.com/liltom-eth/llama2-webui)
- PrivateGPT [(imartinez/privateGPT: Interact privately with your documents using the power of GPT, 100% privately, no data leaks (github.com))](https://github.com/imartinez/privateGPT)
- H2oGPT [(h2oai/h2ogpt: Private Q&A and summarization of documents+images or chat with local GPT, 100% private, Apache 2.0. Supports LLaMa2, llama.cpp, and more. Demo: https://gpt.h2o.ai/ (github.com)](https://github.com/h2oai/h2ogpt)
- vLLM which can be seen as a buffer and shared memory feature that can be used to drastically increase the response time for a model improving the number of tokens the model can generate in response each second.
[(vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention)](https://vllm.ai/)

Some also offer a web UI integrated with a cloud-based language model.

### <a id="loading-an-appropriate-benchmarking-dataset"></a>Load an appropriate benchmarking dataset
Datasets mainly fall under multiple categories which aim to test proficiency of a model at a particular task such as question answering or summarization such Squad and RACE. Hybrid benchmarks such as SuperGlue and LAMBADA were also created to which include a variety of tasks to test the model on to get a more holistic understanding of the a large language model's capabilities. The popularity and relevance of a dataset can be gauged from what the latest foundational models use such as:
- AGIEval (Human standardized tests proficiency) and BigBench Hard (Subset of problems which are yet to surpass Human Level Performance (HLE)) for Orca
- MMLU (Its metrics are agreed upon to indicate proximity to acheiving AGI) - for Llama-2 
We also prefer minimally curated datasets to ensure that the dataset is not biased towards a particular model. 

![image](static/assets/image8.gif)
*This visualization from Google shows how the complexity and score of an LLM can grow within different areas with the number of parameters*

### <a id="selecting-a-relevant-metric-for-evaluation"></a> Selecting a relevant metric for evaluation
The most common metrics used for evaluation are:
- Perplexity
  - Pros:
    - Fast to calculate
    - Useful in Estimating a Language Model’s uncertainty
    - Statistically Robust
  - Cons:
    - Not Accurate for Final Evaluation as it’s possible for a model to have low perplexity but a high error rate.
    - It can be hard to make comparisons across datasets because each dataset has its own distribution of words, and each model has its own parameters.
    - If a newer dataset contains words not present in the training data, the perplexity of the models trained on that data will be artificially inflated.
  - cannot 
- F1 Score
  - Pros:
    - Balanced Evaluation: F1 score considers both precision and recall, providing a balanced evaluation of model performance.
    - Useful for Imbalanced Data: Particularly useful when dealing with imbalanced classes, as it considers false positives and false negatives.
  - Cons:
    - Single Threshold: F1 score requires choosing a specific threshold, which might not be optimal for all applications.
    - Binary Classification Focus: Originally designed for binary classification, it might not be the best fit for all NLP tasks.
- BLEU Score
  - Pros:
    - Language independent
    - Correlates well with human evaluation
  - Cons:
    - Insensitive to Semantic Meaning: BLEU does not capture semantic similarity well and may favor literal translations, penalizing creative and contextually appropriate responses.
    - Reference Dependency: Scores can be highly sensitive to the choice and number of reference texts.
- ROUGE Score
  - Pros:
    - Recall-Oriented: ROUGE emphasizes recall by measuring overlap of n-grams between generated and reference text.
    - Summarization Evaluation: Commonly used for automatic summarization tasks and can offer insights into content overlap.
  - Cons:
    - Limited to N-grams: Like BLEU, ROUGE is sensitive to n-gram overlap and might not capture semantic meaning accurately.
    - Reference Dependency: Similar to BLEU, ROUGE scores are affected by the choice and number of reference summaries.
- Accuracy
  - Pros:
    - Intuitive: Accuracy is easy to understand and interpret, representing the proportion of correct predictions.
    - Applicability: Suitable for classification tasks and when classes are roughly balanced.
  - Cons:
    - Imbalanced Data Issues: Accuracy can be misleading when dealing with imbalanced classes, as it can be high even if the model predominantly predicts the majority class.
    - Doesn't Capture Confidence: Doesn't consider the confidence or certainty of the predictions.
- Human Evaluation
  - Pros:
    - Holistic Assessment: Human evaluation provides a comprehensive assessment of model outputs, considering fluency, coherence, contextuality, and other qualitative aspects.
    - Real-World Relevance: Especially relevant when the ultimate goal is human satisfaction, such as in chatbots or content generation.
  - Cons:
    - Subjectivity: Human evaluation can be subjective and prone to biases, making it challenging to establish consistent benchmarks.
    -Resource-Intensive: Requires human annotators, time, and resources, making it less feasible for large-scale evaluations.

# <a id="observations"></a> Observations


| Bencmark (Higher is Better) | MPT (7B) | Falcon (7B) | LLama-2 (7B) | Llama-2 (13B) | MPT (30B) | Falcon (40B) | Llama-1 (65B) | Llama-2 (70B) |
|:---------------------------:|:--------:|:-----------:|:------------:|:-------------:|:---------:|:------------:|:-------------:|:-------------:|
|           *MMLU*          |   26.8   |     26.2    |     45.3     |      54.8     |    46.9   |     55.4     |      63.4     |    **68.9**   |
|         *TriviaQA*        |   59.6   |     56.8    |     68.9     |      77.2     |    71.3   |     78.6     |      84.5     |     **85**    |
|    *Natural Questions*    |   17.8   |     18.1    |     22.7     |      28.0     |    23.0   |     29.5     |      31.0     |    **33.0**   |
|          *GSM8K*          |    6.8   |     6.8     |     14.6     |      28.7     |    15.2   |     19.6     |      50.9     |    **56.8**   |
|        *HumanEval*        |   18.3   |     N/A     |     12.8     |      18.3     |    25.0   |      N/A     |      23.7     |    **29.9**   |
|         *AGIEval*         |   23.5   |     21.2    |     29.3     |      39.1     |    33.8   |     37.0     |      47.6     |    **54.2**   |
|          *BoolQ*          |   75.0   |     67.5    |     77.4     |      81.7     |    79.0   |     83.1     |    **85.3**   |      85.0     |
|        *HellaSwag*        |   76.4   |     74.1    |     77.2     |      80.7     |    79.9   |     83.6     |      84.2     |    **85.3**   |
|        *OpenBookQA*       |   51.4   |     51.6    |     58.6     |      57.0     |    52.0   |     56.6     |      60.2     |    **60.2**   |
|           *QuAC*          |   37.7   |     18.8    |     39.7     |      44.8     |    41.1   |     43.3     |      39.8     |    **49.3**   |
|        *Winogrande*       |   68.3   |     66.3    |     69.2     |      72.8     |    71.0   |     76.9     |      77.0     |    **80.2**   |


# <a id="tools-overview"></a> Tools Overview
There are many packages that assist in evaluation of Large Language Models (LLMs). We take the best practices available along with our own optimizations to create one-stop method to evaluate LLMs holistically aiming to provide a fair, open, and reproducible benchmark for large model evaluation. Its main features include:

- **Static Evaluations** - Uses OpenCompass package to for coverage on most of popular benchmarking datasets and large language models.
- **Evaluation Levels** - Dataset independent evaluations similar to the one [here](https://lmsys.org/vicuna_eval/).
- **LLM-as-a-judge** - Uses FastChat's LLM-as-a-judge to evaluate your models with MT-bench questions and prompts which is a set of challenging multi-turn open-ended questions for evaluating chat assistants.
- **OpenAI Evals** - Evals is a framework for evaluating LLMs (large language models) or systems built using LLMs as components. It also includes an open-source registry of challenging evals.

#### <a id="running-evals"></a> Running evals
- Learn how to run existing evals: [run-evals.md](https://github.com/openai/evals/blob/main/docs/run-evals.md).
- Familiarize yourself with the existing eval templates: [eval-templates.md](https://github.com/openai/evals/blob/main/docs/eval-templates.md).

This concept allows for multiple levels of evaluating the effectiveness of a large language models.

## <a id="installation"></a> Installation

  To install the `llm_benchmarking_suite` as a PyPi package, use the following pip command:
To install the `llm_benchmarking_suite` package, use the following pip command:

```bash
pip install llm_benchmarking_suite
```

## Get Started

### <a id="prerequisites"></a> Pre-requisites
Run the following to command in a linux machine to check _CUDA toolkit_ and _cuDNN_ is correctly configured. This will not run in macOS or Windows.
```bash
nvidia-smi
nvcc --version
```
### <a id="environment-setup"></a>Environment Setup
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


## <a id="run-static-evaluations"></a> Run static evaluations
1. 
```bash
cd opencompass && python opencompass/run.py configs/eval_demo.py -w outputs/demo
```
2. To view the locally created metrics dashboard start a server at `cd ../static && python3 -m http.server 8000` and go to http://localhost:8000/ in your browser. Alternatively you can also compare you're evaluations with ours at [LLM Benchmarker Suite Leaderbaord](https://llm-evals.formula-labs.com/
).



### <a id="important-suite-tools"></a> Important Suite Tools:
The Suite comprises of tools to help you carry out metrics analysis on large language models in a bunch of different ways to suit your use case.

#### <a id="opencompass"></a> OpenCompass:
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

#### <a id="llm-as-a-judge"></a> FastChat's LLM-as-a-Judge:
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
#### <a id="openai-evals"></a> OpenAI Evals
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

#### Running evals
- Learn how to run existing evals: [run-evals.md](docs/run-evals.md).
- Familiarize yourself with the existing eval templates: [eval-templates.md](docs/eval-templates.md).

This concept allows for multiple levels of evaluating the effectiveness of a large language models.

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

### <a id="datasets"></a> Datasets
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

## <a id="leaderboard"></a> Leaderboard

We provide [LLM Benchmarker Suite Leaderbaord](https://llm-evals.formula-labs.com/
) for community to rank all public models and API models. If you would like to join the evaluation, please provide the model repository URL or a standard API interface to the email address `abhijoy.sarkar@theoremone.co`.


## <a id="dataset-support"></a> Dataset Support

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

## <a id="model-support"></a> Model Support

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

