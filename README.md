<h1 align="center">Wildfire Smoke Detection </h1>

![Smoke Detection](https://img.shields.io/badge/Task-Smoke_Detection-blue)
![Travel Planner](https://img.shields.io/badge/Task-Tool_Use-blue) 
![GPT-4](https://img.shields.io/badge/Model-GPT--4-green) 
![LLMs](https://img.shields.io/badge/Model-LLMs-green)

Code for the Paper "[Exploring the Binary Classification of Wildfire Smoke Through Vision-Language Models](http://arxiv.org/abs/2402.01622)".
<!TODO: change link>

<p align="center">
[<a href="http://arxiv.org/abs/2402.01622">Paper</a>] •
[<a href="https://legacy-www.hpwren.ucsd.edu/FIgLib/">Dataset</a>]
</p>



# TravelPlanner

TravelPlanner is a benchmark crafted for evaluating language agents in tool-use and complex planning within multiple constraints.

For a given query, language agents are expected to formulate a comprehensive plan that includes transportation, daily meals, attractions, and accommodation for each day.

For constraints, from the perspective of real world applications, TravelPlanner includes three types of them: Environment Constraint, Commonsense Constraint, and Hard Constraint. 


## Setup Environment

1. Create a conda environment and install dependency:
```bash
conda create -n wildfire-smoke-detection python=3.9
conda activate wildfire-smoke-detection
pip install -r requirements.txt
```

2. Download the [database](https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view?usp=drive_link) and unzip it to the `TravelPlanner` directory (i.e., `your/path/TravelPlanner`).

## Setup Model
### LLaVA
Setup a LLaVA server on localhost using the official [LLaVA repository](https://github.com/haotian-liu/LLaVA). Simply clone their repository and add the llava_server.py file found in this repository into their root directory and run it.

### PaliGemma
The PaliGemma model is developed by Google, and you can read about it on its [blog](https://ai.google.dev/gemma/docs/paligemma).
```bash
python utils/servers/paligemma_gradio_server.py
```
### Phi3
The Phi3 is a small model developed by Microsoft and you can read more about it [here](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
```bash
python utils/servers/phi3_gradio_server.py
```
### GPT4
Obtain an OpenAI api key and export it as an environment variable
```bash
export OPENAI_API_KEY="your api key"
```

## Running
### Zero-Shot Mode

In the zero-shot mode, the vision-language model is prompted with the entire image directly

```bash
export OUTPUT_DIR=path/to/your/output/file
# We support MODEL in ['gpt-3.5-turbo-X','gpt-4-1106-preview','gemini','mistral-7B-32K','mixtral']
export MODEL_NAME=MODEL_NAME
export OPENAI_API_KEY=YOUR_OPENAI_KEY
# if you do not want to test google models, like gemini, just input "1".
export GOOGLE_API_KEY=YOUR_GOOGLE_KEY
# SET_TYPE in ['validation', 'test']
export SET_TYPE=validation
cd agents
python tool_agents.py  --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME
```
The generated plan will be stored in OUTPUT_DIR/SET_TYPE.

### Sole-Planning Mode

TravelPlanner also provides an easier mode solely focused on testing their planning ability.
The sole-planning mode ensures that no crucial information is missed, thereby enabling agents to focus on planning itself.

Please refer to paper for more details.

```bash
export OUTPUT_DIR=path/to/your/output/file
# We support MODEL in ['gpt-3.5-turbo-X','gpt-4-1106-preview','gemini','mistral-7B-32K','mixtral']
export MODEL_NAME=MODEL_NAME # langfun
export OPENAI_API_KEY=YOUR_OPENAI_KEY
# if you do not want to test google models, like gemini, just input "1".
export GOOGLE_API_KEY=YOUR_GOOGLE_KEY
# SET_TYPE in ['validation', 'test', 'train']
export SET_TYPE=validation
# STRATEGY in ['direct','cot','react','reflexion', 'by_day']
export STRATEGY=direct

cd tools/planner
python sole_planning.py  --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --strategy $STRATEGY
```

## Postprocess

In order to parse natural language plans, we use gpt-4 to convert these plans into json formats. We encourage developers to try different parsing prompts to obtain better-formatted plans.

```bash
export OUTPUT_DIR=../evaluation
export MODEL_NAME=gpt-4-1106-preview
export SET_TYPE=validation
export STRATEGY=direct
export TMP_DIR=.
export EVALUATION_DIR=../evaluation
export MODE=sole-planning
export SUBMISSION_FILE_DIR=./

cd postprocess
python parsing.py  --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --strategy $STRATEGY --tmp_dir $TMP_DIR --mode $MODE

# Then these parsed plans should be stored as the real json formats.
python element_extraction.py  --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --strategy $STRATEGY --tmp_dir $TMP_DIR --mode $MODE

# Finally, combine these plan files for evaluation. We also provide a evaluation example file "example_evaluation.jsonl" in the postprocess folder.
python combination.py --set_type $SET_TYPE --output_dir $OUTPUT_DIR --model_name $MODEL_NAME --strategy $STRATEGY --submission_file_dir $SUBMISSION_FILE_DIR --mode $MODE
```

## Evaluation

We support the offline validation set evaluation through the provided evaluation script. To avoid data contamination, please use our official [leaderboard](https://huggingface.co/spaces/osunlp/TravelPlannerLeaderboard) for test set evaluation.

```bash
export SET_TYPE=validation
export EVALUATION_FILE_PATH=../postprocess/train_gpt-4-1106-preview_direct_sole-planning_submission.jsonl

cd evaluation
python eval.py --set_type $SET_TYPE --evaluation_file_path $EVALUATION_FILE_PATH
```

## Load Datasets

```python
from datasets import load_dataset
# test can be substituted by "train" and "validation".
data = load_dataset('osunlp/TravelPlanner','test')['test']
```

## TODO

- ##### Code

  - [x] Baseline Code

  - [x] Query Construction Code

  - [x] Evaluation Code
  - [x] Plan Parsing and Element Extraction Code

- ##### Environment

  - [x] Release Environment Database
  - [ ] Database Field Introduction

## Contact

If you have any problems, please contact 
[Jian Xie](mailto:jianx0321@gmail.com),
[Kai Zhang](mailto:zhang.13253@osu.edu),
[Yu Su](mailto:su.809@osu.edu)

## Citation Information

If our paper or related resources prove valuable to your research, we kindly ask for citation. 

<a href="https://github.com/OSU-NLP-Group/TravelPlanner"><img src="https://img.shields.io/github/stars/OSU-NLP-Group/TravelPlanner?style=social&label=TravelPanner" alt="GitHub Stars"></a>

```
@article{xie2024travelplanner,
  title={Travelplanner: A benchmark for real-world planning with language agents},
  author={Xie, Jian and Zhang, Kai and Chen, Jiangjie and Zhu, Tinghui and Lou, Renze and Tian, Yuandong and Xiao, Yanghua and Su, Yu},
  journal={arXiv preprint arXiv:2402.01622},
  year={2024}
}
```
