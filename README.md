# Text2Cypher Training Evaluation Results

## Table of Contents

- [Overview](#overview)
- [Techniques Used](#techniques-used)
- [Dataset](#dataset)
- [Evaluation Methodology](#evaluation-methodology)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [References](#references)

## Overview

This document summarizes the evaluation results of the Text2Cypher training process. The goal of this document is to evaluate 4 different Text2Cypher techniques and compare their performance.

## Techniques Used

The first step was to find a base model capable of generating Cypher queries from natural language, that also has < 7B parameters. The model I chose is `Qwen/Qwen2.5-Coder-3B-Instruct`, which performed well for some basic cypher queries.

Following are the various techniques used for the Text2Cypher task, using this model:

- *Base Model Generation*: The base model was used to generate Cypher queries from natural language prompts without any fine-tuning.
- *Structured Chain-of-Thought (CoT) over Base Model*: This technique involved using a structured CoT approach to generate Cypher queries from natural language prompts. The CoT was designed to guide the model in generating more accurate and relevant queries. (without using fine-tuning)
- *Fine-tuned model over question-schema-query pairs*: 
    - This technique involved fine-tuning the base model using a dataset of question-schema-query pairs. The fine-tuning process was designed to improve the model's ability to generate Cypher queries from natural language prompts. (without using CoT or structured outputs)
    - Number of examples used: ~7700
    - Number of epochs: 1
- *Fine-tuned model using GRPO (Group Relative Policy Optimization) for CoT*: 
    - This technique involved fine-tuning the base model using a dataset of question-schema-query pairs, but instead of direct generation, the model was trained to generate reasoning *first*, then the final Cypher query, in a structured manner. This approach was designed to improve the model's ability to generate Cypher queries from natural language prompts by incorporating Chain-of-Thought reasoning into the generation process.
    - Number of examples used: ~15 examples from `neo4j/text2cypher-2024v1` (stopped early due to acceptable performance during training)

## Dataset

For the evaluation, I picked 50 examples from [`neo4j/text2cypher-2024v1`](https://huggingface.co/datasets/neo4j/text2cypher-2024v1). The first 25 examples correspond to the top 25 examples ordered by length of query, considering longer queries to be (mostly) more complex. The next 25 examples correspond to the middle 25 examples ordered by length of query. Importantly, records which did not have a `database_reference_alias` (were null) were excluded from selection.

## Evaluation Methodology

The evaluation techniques used by Neo4j for the Text2Cypher task were:

- *Exact Match*: The generated Cypher query is an exact match to the expected Cypher query.
- *Google Bleu Score*: The generated Cypher query is compared to the expected Cypher query using the Google BLEU score, which measures the similarity between two sequences of text. A higher BLEU score indicates a better match between the generated and expected queries.

However, these are not the most reliable metrics for evaluating the performance of a model for the following reasons:

- *Exact Match*: The exact match metric is too strict, as it does not account for minor variations in the generated query that may still be semantically correct. For example, the order of clauses in a Cypher query can vary without changing the meaning of the query.
- *Google Bleu Score*: The BLEU score is a statistical measure that compares the generated query to the expected query based on n-grams. However, it does not account for the semantic meaning of the queries, and can be affected by minor variations in wording or structure. Additionally, BLEU scores can be misleading when comparing queries of different lengths or structures.

This is why I stuck to *Human Evaluation of Execution Results* as the main evaluation technique. This is because:

- The models consistently get the syntax of the Cypher queries wrong, such as the direction of relationships, use of LENGTH vs SIZE, not considering usage of important cypher functions.
- As the models are also trained on SQL queries, they often mix up SQL and Cypher syntax, generating functionally incorrect queries.
- There may be multiple different unique, most importantly, correct ways to write a Cypher query for the same task. 
- As the technique involved executing the queries, some gold queries, though functionally correct, had no results on execution. Due to this, evaluating the queries based on execution results could not be directly possible (only for 2-5 examples). In this case, directly comparing the generated queries with the expected queries, by a human evaluator, was the best option.

The 4 techniques were then used to generate 5 outputs for each example, which were then evaluated as described above. A *Best of 5* approach was used to select the best output for each example, which was then compared to the expected output. A technique *successfully passed* an example if at least one of the 5 outputs was correct. 

## Results

| Technique | Success Rate | Passing Example Id's |
|-----------|--------------|----------------------|
| Base Model Generation | 20% | [8, 26, 29, 30, 32, 33, 36, 41, 42, 46] |
| Structured CoT over Base Model | 20% | [1, 26, 27, 29, 31, 32, 36, 38, 41, 46] |
| Fine-tuned model over question-schema-query pairs | *28%* | [26, 27, 29, 30, 31, 32, 33, 34, 36, 39, 41, 42, 46] |
| Fine-tuned model using GRPO for CoT | **48%** | [1, 8, 13, 17, 22, 26, 27, 29, 30, 31, 32, 33, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 49] |

## Conclusion

- The results of the evaluation indicate that the fine-tuned model using GRPO for CoT outperformed the other techniques, achieving a success rate of *48%*. This suggests that *incorporating Chain-of-Thought reasoning* into the generation process can *significantly improve* the model's ability to *generate Cypher queries from natural language* prompts and, very importantly, for *other similar tasks*.
- This is a promising result, as it indicates that *smaller, more efficient models* can be trained to *perform well* on *complex tasks* like Text2Cypher. The other techniques, while still showing some success, did not perform as well as the fine-tuned model using GRPO for CoT.
- As GRPO is the RL technique being used, that requires comparatively lesser data, it paves the way for tasks for which there is little training data available. 
- The GRPO-tuned model was also capable of reliably generating syntactically correct Cypher queries, which is a significant improvement over the base model and the other techniques.
- It also showcased greater generalization capabilities, as it was able to generate correct queries for examples that were not part of the training set.

## Future Work

- There is a dire need for more robust, automated evaluation techniques for the Text2Cypher task. The current evaluation methods are not sufficient to accurately assess the performance of the models, and more advanced techniques are needed to provide a more comprehensive evaluation.
- A more complete training methodology is needed, which first *teaches* the model Cypher language, it's syntax, functions and so on (if it does not already know it), and then trains it on the Text2Cypher task, using the GRPO technique.
- The current evaluation was limited to 50 examples, and a larger dataset should be used for future evaluations to provide a more comprehensive assessment of the model's performance.
- There is a need for more diverse and complex examples, that allow executing them on a real online database, with tangible results. This would provide a more comprehensive evaluation of the model's performance and its ability to generate Cypher queries from natural language prompts.
- This evaluation was limited to a single model, and future evaluations should include a wider range of models to provide a more comprehensive assessment of the performance of different techniques.

## References

- [Qwen/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)
- [neo4j/text2cypher-2024v1](https://huggingface.co/datasets/neo4j/text2cypher-2024v1)
- [Google BLEU Score](https://en.wikipedia.org/wiki/BLEU)
- [GRPO](https://arxiv.org/abs/2402.03300)
- [Direct Generation Training Data](https://huggingface.co/datasets/tomasonjo/text2cypher-gpt4o-clean)
- [GRPO Training Data](https://huggingface.co/datasets/tomasonjo/text2cypher-gpt4o-rl)
- [Sample GRPO Training Code](https://colab.research.google.com/drive/1hRIql1PnPf2mN52_aPUbaQ9S0vFpdCdh?usp=sharing)
- [Sample Fine-tuning Code (w/o reasoning)](https://www.kaggle.com/code/gurveersinghvirk/sft-text2cypher)
- [Code for generating Outputs for Eval](https://www.kaggle.com/code/gurveersinghvirk/text2cypher-eval)
- [Evaluation Code](https://colab.research.google.com/drive/1iLrkwKRncWuBl9_WM9Fw5fJG8FVI7Cpp?usp=sharing)
- [Evaluation Data](https://huggingface.co/datasets/Gurveer05/text2cypher-small/tree/90597dd368ae626241f093919ac81e05a4a9147f)