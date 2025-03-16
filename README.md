# 2024 NTU ADL HW1
This repository contains coursework from National Taiwan University CSIE5431: Applied Deep Learning.
* Student: I-Hsin Chen
* Affiliation: Graduate Institute of Networking and Multimedia, National Taiwan University

## Environment Setup
```
conda create --name adl-hw1 python=3.10
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4 transformers==4.44.2 datasets==2.21.0 accelerate==0.34.2 evaluate matplotlib tqdm pandas scikit-learn
```

## Train
```
bash train.sh ./path/to/context.json ./path/to/train.json ./path/to/valid.json
```
The output model, tokenizer, and data would be save at `--output-dir` configured in `train.sh`.

## Inference

1. Download model
```
bash ./download.sh
```
The multiple choice model and question answering model would be located at `./models`

2. Run testing data
```
bash run.sh ./path/to/context.json ./path/to/test.json ./path/to/prediction.csv
```
* In this step, a `mc_pred.json` file would be generate as the temporary prediction output from multiple choice model.
* The final prediction result would be saved at `./path/to/prediction.csv`.
