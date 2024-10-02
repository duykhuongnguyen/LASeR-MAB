# LASeR (**L**earning to **A**daptively **Se**lect **R**eward Models)

Code for the paper [LASeR: Learning to Adaptively Select Reward Models with Multi-Arm Bandits]()

[Duy Nguyen](https://duykhuongnguyen.github.io/)\*, [Archiki Prasad](https://archiki.github.io/)\*, [Elias Stengel-Eskin](https://esteng.github.io/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/) (\*equal contribution).

## Abstract
Reward Models (RMs) play a crucial role in aligning large language models (LLMs) with human preferences, enhancing their performance by ranking outputs during inference or iterative training. However, the degree to which an RM generalizes to new tasks is often not known a priori. For instance, some RMs may excel at scoring creative writing, while others specialize in evaluating math reasoning. Therefore, using only one fixed RM while training LLMs can be suboptimal. Moreover, optimizing LLMs with multiple RMs simultaneously can be prohibitively computationally-intensive and challenging due to conflicting signals from different RMs, potentially degrading performance. To address these challenges, we introduce LASeR (Learning to Adaptively Select Rewards), which iteratively trains LLMs using multiple RMs, selecting and utilizing the most well-suited RM for each instance to rank outputs and generate preference data, framed as a multi-armed bandit problem. Our empirical results on commonsense and math reasoning tasks demonstrate that LASeR can boost iterative LLM optimization by optimizing for multiple RMs, improving the absolute average accuracy of Llama-3-8B over three datasets by 2.67% over training with ensemble RM scores while also showing superior training efficiency (e.g., a 2x speedup). Moreover, on WildChat, a benchmark of instruction-following prompts in open-form generation, we find that using Llama-3-8B LASeR leads to a 71.45% AlpacaEval win rate over sequentially optimizing multiple RMs. Extending to long-context generation tasks, we find that on Llama-3-8B, LASeR achieves an average improvement of 2.64 F1 points on single-document QA tasks and 2.42 F1 points on multi-document QA over random RM selection when used with best-of-n sampling. Our analysis shows that LASeR is robust to noisy rewards and generalizes to multiple settings. Finally, we demonstrate that LASeR's RM selection changes depending on the underlying task or instance and we verify the presence of conflicting preferences from multiple RMs that can be mitigated using LASeR.

![image](assets/bandit_overall.png)


## Installation
This project is built on Python 3.10.11. All dependencies can be installed via:

`pip install -r requirements.txt`

## Project Directory Structure
The project directory is as follows:

```
scripts/
├── dataset/
    ├── strategyqa/
    ├── gsm8k/
    └── mmlu/
├── model/
    ├── __init__.py
    └── response_generator.py
├── utils/
    ├── __init__.py
    ├── config_loader.py
    ├── dataset_manager.py
    ├── linucb.py
    ├── llm_trainer.py
    ├── preference_pair_generator.py
    └── reward_model.py
├── config.yaml
├── train_and_infer.py
├── run_training.sh
└── run_training_all.sh
```

## Running LASeR

### Run LASeR on reasoning tasks
Run LASeR on one dataset, for example StrategyQA:

```bash 
cd scripts
bash run_training.sh strategyqa
```

Run all datasets 
```bash
cd scripts
bash run_training_all.sh
```

You can change the training setup in ```scripts/config.yaml```

### Run LASeR on instuction-following tasks
Coming soon

### Run LASeR on long-context understanding tasks
Coming soon

## Citation
```

```