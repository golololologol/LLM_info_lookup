import gradio as gr
import json
from transformers import AutoTokenizer
import hashlib

model_history = {}

try:
    with open("model_history.json", "r") as file:
        model_history = json.load(file)
except FileNotFoundError:
    pass

def try_load_tokenizer(model_path: str) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    except:
        raise ValueError(f"Tokenizer for {model_path} could not be loaded")
    
    return tokenizer

def get_tokenizer_sha(tokenizer = None, model_path="") -> str:
    tokenizer = try_load_tokenizer(model_path) if tokenizer == None else tokenizer

    base_tokens = sorted(tokenizer._tokenizer.get_vocab(with_added_tokens=False).keys())
    tokenizer_sha = hashlib.sha256("".join(base_tokens).encode()).hexdigest()
    return tokenizer_sha

def get_vocab_family(tokenizer=None, model_path="") -> str:
    tokenizer = try_load_tokenizer(model_path) if tokenizer == None else tokenizer

    tokenizer_sha = get_tokenizer_sha(tokenizer)

    sha_to_family = {
        "943d2a9918de54ad09e974a6d834ba44191fa98210c1414a822338fcf3f54038": "mistral",
        "d1e4f1ef2bea057b346dd69244790543bec0dca751118e523d62ee6ae30b6036": "llama_1|2",
        "7f1739ca602e925d6c93e42c0d4d4f6f607169f5b016d5d322e4ec42a1c7c563": "llama_3",
        "55b3930bf06e1139760a921273642322f38c2165620b34f1e6d521a7bc431c90": "command-r",
        "24c4ea5aecffe896ab422b984ec1e4598d5aa565417432271ae752594a71e94e": "dbrx",
        "99fe18abd82f35c8ef2a82b1277059dbe16d0a033fecdfa7726a3a6f574a3866": "gpt2",
        "5520ff09e574cbb231f2baf283bd1a2e358b5d747c1a6aaa6402ee03991a409d": "qwen2",
        "21d0e3b8f0c9f89b7b6f9a4c5b72b6ba2f70e7805460ca4cbd4b9c55d29f31fc": "gptneox",
        "20c2a98cfdd0ae39798458b5a6dc4838d1b700a0ec074525654a57bfa3088648": "gemma",
        "4088308e46cd5596c6c320bd102e48ee33b03d42d12447ccce8beafe4310bfa3": "yi",
        "2e7d13c6f9a9825b1dfeb645fe3130b118e4c119bdf0460be06bd7e2d7660728": "deepseek",
        "62947c306f3a11187ba2a4a6ea25de91ce30c5724e6b647a1d6f0f8868217ead": "deepseek_1.5",
        "0d35d803249ab1f34c99f8a10a562fc433134cca4fbd566dcd6ca61c4a857b04": "T5"
    }

    vocab_family = sha_to_family.get(tokenizer_sha, "Unknown") # type: ignore
    return vocab_family

def save_history():
    with open("model_history.json", "w") as file:
        json.dump(model_history, file, indent=4)

def update_history_display():
    entries = []
    for key, value in model_history.items():
        if "error" in value:
            entries.append(f"{key}: {value['error']}")
            continue

        vocab_family = value.get("vocab_family", "Unknown")
        tokenizer_sha = value.get("tokenizer_sha", None)

        if vocab_family == "Unknown":
            entries.append(f"{key}: {vocab_family}, Tokenizer SHA: {tokenizer_sha}, Likely vocab family: {value['tokenizer_class']}")
        else:
            entries.append(f"{key}: {vocab_family}")

    return "\n".join(entries)

def get_vocab_info(request, branch="main", show_special_tokens=False, use_local_cache=True):
    if not request.strip():
        return None, None, None

    parts = request.split('/')
    if 'tree' in parts:
        branch_index = parts.index('tree') + 1
        branch = parts[branch_index]
        model_name = '/'.join(parts[-4:-2])  # model_creator/model_name
    else:
        model_name = '/'.join(parts[-2:])  # model_creator/model_name
        if '(' in model_name:
            # Splitting out branch from model name if formatted like 'model_name (branch)'
            model_name, branch_part = model_name.split(' (')
            branch = branch_part[:-1]  # Strip the closing parenthesis

    model_key = f"{model_name} ({branch})"

    # Check the cache with the formatted key
    if model_key in model_history and use_local_cache:
        vocab_info = model_history[model_key]
        return vocab_info, model_name, branch

    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=branch, trust_remote_code=True, use_fast=True)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=branch, trust_remote_code=True, use_fast=False)

        test_input = [
            {"role": "user", "content": "{UserContent}"},
            {"role": "assistant", "content": "{AIContent}"}
        ]

        vocab_family = get_vocab_family(tokenizer)
        vocab_info = {
            "vocab_family": vocab_family,
            "tokenizer_sha": get_tokenizer_sha(tokenizer),
            "bos": tokenizer.bos_token, 
            "eos": tokenizer.eos_token,
            "pad": tokenizer.pad_token, 
            "unk": tokenizer.unk_token,
            "bos_id": tokenizer.bos_token_id, 
            "eos_id": tokenizer.eos_token_id,
            "pad_id": tokenizer.pad_token_id, 
            "unk_id": tokenizer.unk_token_id,
            "vocab_size": tokenizer.vocab_size,
            "full_vocab_size": tokenizer.vocab_size + tokenizer.added_tokens_encoder.keys().__len__(),
            "default_prompt_format": tokenizer.chat_template == None,
            "prompt_format": tokenizer.apply_chat_template(test_input, tokenize=False, add_generation_prompt=True),
            "tokenizer_class": tokenizer.__class__.__name__ if hasattr(tokenizer, "__class__") else "Unknown",
            "all_special_tokens": tokenizer.added_tokens_encoder
        }

        model_history[model_key] = vocab_info
        if use_local_cache:
            save_history()
    except Exception as e:
        vocab_info = {"error": str(e)}
    
    return vocab_info, model_name, branch

def find_model_info(model_input, show_special_tokens, use_local_cache):
    requests = model_input.split("\n")
    results = []
    for request in requests:
        
        vocab_info, model_name, branch = get_vocab_info(request, show_special_tokens=show_special_tokens, use_local_cache=use_local_cache)

        if vocab_info is None:
            continue

        result_str = f"{model_name} ({branch}): {vocab_info.get('vocab_family', 'Unknown')}"

        if "error" in vocab_info:
            result_str += f', error: {vocab_info["error"]}'
            results.append(result_str)
            continue
        elif vocab_info.get('vocab_family', 'Unknown') == "Unknown":
            result_str += f'\nTokenizer SHA: {vocab_info["tokenizer_sha"]}'
            result_str += f"\nLikely vocab family: {vocab_info['tokenizer_class']}"

        result_str += f"\nBOS: {vocab_info['bos']} id: {vocab_info['bos_id']}"
        result_str += f"\nEOS: {vocab_info['eos']} id: {vocab_info['eos_id']}"
        result_str += f"\nPAD: {vocab_info['pad']} id: {vocab_info['pad_id']}"
        result_str += f"\nUNK: {vocab_info['unk']} id: {vocab_info['unk_id']}"
        result_str += f"\nBase Vocab Size: {vocab_info['vocab_size']}, Full Vocab Size: {vocab_info['full_vocab_size']}"
        prompt_format = vocab_info.get('prompt_format', None)
        if vocab_info['default_prompt_format']:
            result_str += f"\n\n(Likely incorrect) Prompt Format:\n{prompt_format if prompt_format else 'Unknown'}"
        else:
            result_str += f"\n\nPrompt Format:\n{prompt_format if prompt_format else 'Unknown'}"

        if show_special_tokens:
            if vocab_info.get('all_special_tokens', {}) == {}:
                result_str += f"\n\nNo Special Tokens Could Be Fetched"
            else:
                result_str += f"\n\nAll Special Tokens:"
                for token, id in vocab_info['all_special_tokens'].items():
                    result_str += f"\n{token} id: {id}"
        
        results.append(result_str)

    history_display = update_history_display()
    return ('\n' + '-' * 130 + '\n').join(results), history_display

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=5):
            model_input = gr.Textbox(label="Model Name", show_label=False, placeholder="author/model_name", info="Accepts author/model, and full URLs to model repos, e.g. huggingface.co/author/model")
            with gr.Row():
                show_special_tokens_toggle = gr.Checkbox(label="Show all special tokens", value=False, interactive=True)
                use_local_cache = gr.Checkbox(label="Use local cache", value=True, interactive=True)

        submit_button = gr.Button(value="Retrieve Model Info", scale=4)
    with gr.Row():
        model_info_output = gr.Textbox(label="Model Information", interactive=False)
        model_history_output = gr.Textbox(label="Model History", interactive=False, value=update_history_display())

    submit_button.click(
        fn=find_model_info, 
        inputs=[model_input, show_special_tokens_toggle, use_local_cache], 
        outputs=[model_info_output, model_history_output]
    )

demo.launch(debug=True)
