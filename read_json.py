import json

import importlib


def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

# Load the JSON config file
config = load_config('config.json')

train_steps = config['Train_steps']

root_dir = config['folder']['root']

data_path = root_dir + config['folder']['data_path']
output_path = root_dir  + config['folder']['output_path']

# To expand code for each step, easy for debugging
for step in train_steps:
    if step == "step1":
        continue
        module = importlib.import_module(step)

        step_cfg = config['step1']
        input_data = step_cfg['input_data']
        sample_index_file = step_cfg['sample_index'] 
        output_model = step_cfg['output_model']

        module.train(data_path, output_path, input_data, sample_index_file, output_model)

    elif step == 'step2':
        continue
        module = importlib.import_module(step)

        step_cfg = config['step2']
        input_data = step_cfg['input_data']
        sample_index_file = step_cfg['sample_index'] 
        pretrained_model = step_cfg['pretrained_model']
        output_model = step_cfg['output_model']
        synthetic_data = step_cfg['synthetic_data']

        module.train(data_path, output_path, input_data, sample_index_file, 
          pretrained_model, output_model, synthetic_data)

    elif step == 'step3':
        #continue
        module = importlib.import_module(step)

        step_cfg = config['step3']
        input_data = step_cfg['input_data']
        #sample_index_file = step_cfg['sample_index'] 
        pretrained_model = step_cfg['pretrained_model']
        output_model = step_cfg['output_model']
        synthetic_data = step_cfg['synthetic_data']

        module.train(data_path, output_path, input_data, #sample_index_file, 
          pretrained_model, output_model, synthetic_data)
    elif step == 'step4':
        #continue
        module = importlib.import_module(step)

        step_cfg = config['step4']
        input_data = step_cfg['input_data']
        sample_index_file = step_cfg['sample_index'] 
        pretrained_model = step_cfg['pretrained_model']
        output_model = step_cfg['output_model']
        synthetic_data = step_cfg['synthetic_data']

        module.train(data_path, output_path, input_data, sample_index_file, 
          pretrained_model, output_model, synthetic_data)

    elif step == 'step5':
        module = importlib.import_module(step)

        step_cfg = config['step5']
        input_data = step_cfg['input_data']
        sample_index_file = step_cfg['sample_index'] 
        pretrained_model = step_cfg['pretrained_model']
        output_model = step_cfg['output_model']
        synthetic_data = step_cfg['synthetic_data']
        fluxtower_inputs = step_cfg['fluxtower_inputs']
        fluxtower_observe = step_cfg['fluxtower_observe']

        module.train(data_path, output_path, input_data, sample_index_file, 
          pretrained_model, output_model, synthetic_data, fluxtower_inputs, fluxtower_observe)
    else:
        print(f" Wrong step {step}")

