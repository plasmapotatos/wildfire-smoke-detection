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


## Setup Environment

1. Create a conda environment and install dependency:
```bash
conda create -n wildfire-smoke-detection python=3.9
conda activate wildfire-smoke-detection
pip install -r requirements.txt
```

2. Download the [database](https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view?usp=drive_link) and unzip it to the `wildfire-smoke-detection` directory (i.e., `your/path/wildfire-smoke-detection`).
3. Process the data.
```bash
python scripts/process_data.py
```
## Setup Model
Choose a model to test. We recommend the Phi3 model as it is quite small and can easily be run locally. The following scripts will set up a local server to run the model.
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

In the zero-shot mode, the vision-language model is prompted with the entire image directly.

```bash
# we support "llava", "paligemma", "phi3", and "gpt4".
export MODEL_NAME=MODEL_NAME

python agents/zeroshot_agent.py
```
The output results will be stored in the series_results/MODEL_NAME/tiled/1x1 folder.
### Image Tiling Mode

In the image tiling mode, the image is first tiled into a number of tiles of equal size, which are then each fed separately into the vision-language model.

```bash
# we support "llava", "paligemma", "phi3", and "gpt4".
export MODEL_NAME=MODEL_NAME
export NUM_ROWS=NUM_ROWS # Default is 4 rows
export NUM_COLS=NUM_COLS # Default is 4 columns

python agents/image_tiling_agent.py
```
The output results will be stored in the results/MODEL_NAME/tiled/NUM_ROWSxNUM_COLS folder.

### Horizon Tiling Mode
In the horizon tiling mode, the image is first tiled across the middle with a number of tiles, which are then each fed separately into the vision-language model.
```bash
# we support "llava", "paligemma", "phi3", and "gpt4".
export MODEL_NAME=MODEL_NAME
export DIST_ABOVE=DIST_ABOVE # Default is 400 pixels above middle
export DIST_BELOW=DIST_BELOW # Default is 300 pixels below middle
export TILE_NUMBER=TILE_NUMBER # Number of tiles to be spaced across middle, default 5
export NUM_TILES=NUM_TILES # Number of tiles that can be spaced across the middle, used to calculate the tile width, default is 4 to create overlap

python agents/horizon_agent.py
```
The output results will be stored in the results/MODEL_NAME/horizon/TILE_NUMBERxNUM_TILES folder.
## Evaluation

We support the offline validation set evaluation through the provided evaluation script.

```bash
# we support "llava", "paligemma", "phi3", and "gpt4".
export MODEL_NAME=MODEL_NAME
export MODE=MODE #can be either tiled or horizon, if you are running zero-shot, put tiled
export RESULTS_PATH=RESULTS_PATH # path to results folder, ex: results/gpt4/tiled/4x4

python evaluation/eval_series.py # saves statistics file into the results_path folder

export FILE_PATH=FILE_PATH # path to statistics file, ex: "results/gpt4/tiled/4x4/series_evaluation_stats.json"

python evaluation/process_stats.py
```
The processed statistics will be saved to the processed_results.json file.

## Contact

If you have any problems, please contact 
[Timothy Wei](mailto:timswei@gmail.com).

## Citation Information

If our paper or related resources prove valuable to your research, we kindly ask for citation. 

Paper under review!
