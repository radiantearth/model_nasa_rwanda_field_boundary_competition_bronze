#!/usr/bin/env python
# coding: utf-8

# ----------------------------------------

input_dir      = "/opt/radiant/data/input"
output_dir     = "/opt/radiant/data/output"
workspace_dir  = "/opt/radiant/workspace"
models_dir     = "/opt/radiant/models"


# ----------------------------------------

import json
from inference import run_inference

model_name = "m1"
encoder = 'efficientnet-b0' # 'timm-efficientnet-l2'
decoder = 'UnetPlusPlus'
encoder_weights = 'imagenet' # 'noisy-student'


with open(f'{models_dir}/models.json') as f:
	models_json = json.load(f)

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
	print("-"*40)
