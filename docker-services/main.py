#!/usr/bin/env python
# coding: utf-8

# ----------------------------------------

input_dir      = "/opt/radiant/data/input"
output_dir     = "/opt/radiant/data/output"
workspace_dir  = "/opt/radiant/workspace"
models_dir     = "/opt/radiant/models"


# ----------------------------------------
import numpy as np 
import pandas as pd 
import os

import json
from inference import run_inference

model_name = "m1"
encoder = 'efficientnet-b0' # 'timm-efficientnet-l2'
decoder = 'UnetPlusPlus'
encoder_weights = 'imagenet' # 'noisy-student'

# Meta
with open(f'{models_dir}/models.json') as f:
	models_json = json.load(f)

# DL
output_filenames = []
for model_name in models_json.keys():
	print("-"*40)
	print(f"Running {model_name} ... ")
	model_info = models_json[model_name]

	encoder = model_info["encoder"]
	decoder = model_info["decoder"]
	encoder_weights = model_info["encoder_weights"]

	output_filename = run_inference(
		model_name=model_name,
		encoder=encoder,
		decoder=decoder,
		encoder_weights=encoder_weights,
	)
	output_filenames.append(output_filename)
	print("-"*40)

# Ensemble
dsub = [
    pd.read_csv(output_filename) for output_filename in output_filenames
]

n = len(dsub)
sub = dsub[0].copy()

sub.label = 0
pred_models = []
for d in dsub:
    pred_models.append(d.label.apply(lambda x: int(x >= 0.27)))

sub.label = np.array(pred_models).sum(0)
sub.label = sub.label.apply(lambda x: int(x >= len(pred_models)/2))

output_filename = f"{output_dir}/submission.csv"
sub.to_csv(output_filename, index=False)
sub.label.sum(), sub.label

print(f"Final output is {output_filename} !!!")