# LLM_info_lookup

A straightforward Gradio interface to retrieve key information about PyTorch LLMs.

## Features

- Retrieves and displays model vocab information including special tokens.
- Pulls number of layers, hidden size, and more from the model's config.
- Shows vocabulary family for a range of known tokenizer SHA signatures.
- Provides functionality to use local cache for faster access.
- Customizable model retrieval using branches.

## Installation

1. Clone this repository `git clone https://github.com/golololologol/LLM_info_lookup`.
2. Run `launch.bat` or `launch.sh` depending on your OS.
3. Now you're all set to use it.

## How to Use

1. Run `launch.bat` or `launch.sh`.
2. Go to `http://127.0.0.1:7860` in your browser.
3. Enter the address to the model in the textbox.
4. Click "Retrieve Model Info", and enjoy knowing what the hell that one new model is.

## UI

![image](https://github.com/golololologol/LLM_info_lookup/assets/50058139/c9c52a93-ac2c-48bd-a204-26c5e9b0ad9d)

## Input Formatting

- model_creator/model_name 
- model_creator/model_name (branch)
- model_creator/model_name (branch): vocab_family

Or stick the whole URL in there if you’re lazy:
- https://huggingface.co/model_creator/model_name
- https://huggingface.co/model_creator/model_name/tree/branch

(Not specifying a branch will default to main)

## Contributions

If you want to contribute to this mess, feel free. Just fork the repo, make your changes, and submit a pull request.
