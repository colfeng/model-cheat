# Model Cheating Detection
This code is created for model cheat detection in Shared Task FinLLM of FinNLP AgentScen IJCAl-2024 (https://sites.google.com/nlg.csie.ntu.edu.tw/finnlp-agentscen/shared-task-finllm).

## Introduction

To measure the risk of data leakage from the test set used in the training of a model, the Model Cheating, we have developed a new metric called the Data Leakage Test (DLT), building on existing research [1].

The DLT calculates the difference in perplexity of the large language models (LLMs) on both the training and test data to determine its data generation tendencies. Specifically, we separately input the training set and the test set into the LLMs, and calculate the perplexity on the training set (ppl-on-train) and the perplexity on the test set (ppl-on-test). The DLT value is then computed by subtracting the ppl-on-train from the ppl-on-test. A larger difference implies that the LLM is less likely to have seen the test set during training compared to the training set and suggests a lower likelihood of the model cheating. Conversely, a smaller difference implies that the LLM is more likely to have seen the test set during training and suggests a higher likelihood of the model cheating.

In the detection process, we will calculate the DLT values for some LLMs to establish a reference baseline of Model Cheating, and minimize the impact of generalization on the metric.


## Run

The environmental preparation is the same as FinBen [2] (https://github.com/The-FinAI/PIXIU).

Obtain the DLT of model cheat detection:

```bash
python main.py \
    --model_path "model_path" \
    --train_data_path "train_data_path" \
    --test_data_path "test_data_path" \
    --save_path "save_path" \
    --device "cuda" 
```

Parameter Description

- `model_path`: The path that saves the model
- `train_data_path`: The path that saves the train data of one task
- `test_data_path`: The path that saves the test data of one task (the same task of train data)
- `save_path`: Result saving path
- `device`: cuda or cpu

## Reference
[1] Skywork: A More Open Bilingual Foundation Model (https://arxiv.org/pdf/2310.19341)

[2] The FinBen: An Holistic Financial Benchmark for Large Language Models (https://arxiv.org/pdf/2402.12659)