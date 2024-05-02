import gradio as gr
import json
import os
from transformers import AutoTokenizer, AutoConfig
import hashlib

model_history = {}
model_history_path = "model_history.json"

try:
    with open(model_history_path, "r") as file:
        model_history = json.load(file)
except:
    pass

def clear_history():
    global model_history
    model_history.clear()
    if os.path.exists(model_history_path):
        save_history()
    history_display = update_history_display()
    return history_display

def try_load_tokenizer(model_path: str) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    except:
        raise ValueError(f"Tokenizer for {model_path} could not be loaded")
    return tokenizer

def get_tokenizer_sha(tokenizer = None, model_path="") -> str:
    tokenizer = try_load_tokenizer(model_path) if tokenizer == None else tokenizer

    all_tokens = tokenizer.get_vocab().keys()
    added_tokens = tokenizer.get_added_vocab().keys()
    base_tokens = sorted(set(all_tokens) - set(added_tokens))
    tokenizer_sha = hashlib.sha256("".join(base_tokens).encode()).hexdigest()
    return tokenizer_sha

def get_vocab_family(tokenizer=None, model_path="") -> str:
    tokenizer = try_load_tokenizer(model_path) if tokenizer == None else tokenizer

    tokenizer_sha = get_tokenizer_sha(tokenizer)

    sha_to_family = {
        "154a07d332d0466ce54d5e83190930dc872c95777c493653c48d6b6b01891377": "mistral",
        "88dfafd1e6cd6fc3cf71600f1c8590ec6b457263267d801636320000a6f687e3": "llama_1|2",
        "7f1739ca602e925d6c93e42c0d4d4f6f607169f5b016d5d322e4ec42a1c7c563": "llama_3",
        "748d7e81288e3b759c4458794fee17f546a07e08e70832d2f04486cd1fc76121": "command-r",
        "12c2a843415f76681408ba2645bb7ea68fc00e864afbfa408193159f847fbe4b": "dbrx",
        "f41d538d54aa627ad2d1a7d853759ac104b94775a5de51e6e6e6e112fb32c1de": "gpt2",
        "e59f183b9781a484efd580deda830bd952f7d3694c4648d41d295284a48f2945": "codeqwen",
        "5520ff09e574cbb231f2baf283bd1a2e358b5d747c1a6aaa6402ee03991a409d": "qwen_1.5",
        "347b0481e0536ab58d630b461c8bbb9dceb8b1a3ff6b9250c73bcd423e64fa71": "gptneox",
        "9eb38d83274ea4aac4d1a037331c7b9f51ee5a74bc2ff9d85f3c1b7a944f2fd0": "gemma",

        "b6f82ad160f599b1dd5bec8987acb5a316423d04de3123fa1eb0c8f1ba7f5568": "gemma", # This is stupid, yes, but that's because this:
                                                                                     # https://huggingface.co/alpindale/gemma-7b gemma's tokenizer is fucked,
                                                                                     # and it doesn't treat "<start_of_turn>" and "<end_of_turn>" as special tokens,
                                                                                     # so they end up in the base vocab, and i can't do anything about it 
                                                                                     
        "f6556674148d92703237bab474c2cf220255926e7b6811e526b072d0ed086beb": "yi",
        "2e7d13c6f9a9825b1dfeb645fe3130b118e4c119bdf0460be06bd7e2d7660728": "deepseek",
        "62947c306f3a11187ba2a4a6ea25de91ce30c5724e6b647a1d6f0f8868217ead": "deepseek_1.5",
        "94c18f1464d6aeb4542dff2fb8dc837e131e39853d86707eea683470c7344480": "T5",
        "cabd41803ba4aa362c59603aa9fedd80d8eab202708beccce9f4e1e0b58eaf3f": "codellama",
        "c2ed819dc3c535a3a64a10d492a39baa87b9cc7aa0a2c72adecc1b31e3e1b544": "jamba"
    }

    vocab_family = sha_to_family.get(tokenizer_sha, "Unknown") # type: ignore
    return vocab_family

def save_history():
    global model_history
    with open(model_history_path, "w") as file:
        json.dump(model_history, file, indent=4)

def update_history_display():
    global model_history
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

def collect_info(tokenizer, config):
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
        "all_special_tokens": tokenizer.added_tokens_encoder,
        "architecture": ", ".join(getattr(config, "architectures", ["Unknown"])),
        "activation_function": getattr(config, "hidden_act", "Unknown"),
        "num_layers": getattr(config, "num_hidden_layers", "Unknown"),
        "num_heads": getattr(config, "num_attention_heads", "Unknown"),
        "hidden_size": getattr(config, "hidden_size", "Unknown"),
        "intermediate_size": getattr(config, "intermediate_size", "Unknown"),
        "num_kv_heads": getattr(config, "num_key_value_heads", "Unknown"),
        "max_position_embeddings": getattr(config, "max_position_embeddings", "Unknown"),
        "torch_dtype": getattr(config, "torch_dtype", "Unknown"),
        "model_type": getattr(config, "model_type", "Unknown")
    }
    return vocab_info

def get_vocab_info(request, branch="main", use_local_cache=True):
    if not request:
        return None, None, None
    
    if "?not-for-all-audiences" in request:
        request = request.split("?not-for-all-audiences")[0]

    parts = request.split('/')
    if 'tree' in parts:
        branch_index = parts.index('tree') + 1
        branch = parts[branch_index]
        model_name = '/'.join(parts[-4:-2])  # model_creator/model_name
    else:
        model_name = '/'.join(parts[-2:])  # model_creator/model_name and maybe (branch) and also maybe : vocab_family
        if '(' in model_name:
            # Splitting out branch from model name if formatted like 'model_name (branch)'
            model_name, branch_part = model_name.split(' (')
            if "): " in branch_part:
                branch = branch_part.split("): ")[0]
            else:
                branch = branch_part[:-1]  # Strip the closing parenthesis

    model_key = f"{model_name} ({branch})"

    # Check the cache with the formatted key
    if model_key in model_history and use_local_cache:
        vocab_info = model_history[model_key]
        return vocab_info, model_name, branch

    try:
        config = AutoConfig.from_pretrained(model_name, revision=branch, trust_remote_code=True)
        try: 
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=branch, trust_remote_code=True, use_fast=True)
            vocab_info = collect_info(tokenizer, config)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=branch, trust_remote_code=True, use_fast=False)
            vocab_info = collect_info(tokenizer, config)

        model_history[model_key] = vocab_info
        if use_local_cache:
            save_history()
    except Exception as e:
        vocab_info = {"error": str(e)}
    
    return vocab_info, model_name, branch

def find_model_info(raw_input, show_special_tokens, use_local_cache):
    requests = raw_input.split("\n")
    tokenizer_results = []
    config_results = []

    for request in requests:
        
        vocab_info, model_name, branch = get_vocab_info(request.strip(), use_local_cache=use_local_cache)

        if vocab_info is None:
            continue

        tokenizer_str = f"{model_name} ({branch}): {vocab_info.get('vocab_family', 'Unknown')}"

        if "error" in vocab_info:
            tokenizer_str += f', error: {vocab_info["error"]}'
            tokenizer_results.append(tokenizer_str)
            config_results.append(tokenizer_str)
            continue

        # Tokenizer info
        if vocab_info.get('vocab_family', 'Unknown') == "Unknown":
            tokenizer_str += f'\nTokenizer SHA: {vocab_info["tokenizer_sha"]}'
            tokenizer_str += f"\nLikely vocab family: {vocab_info['tokenizer_class']}"

        tokenizer_str += f"\nBOS: {vocab_info['bos']} id: {vocab_info['bos_id']}"
        tokenizer_str += f"\nEOS: {vocab_info['eos']} id: {vocab_info['eos_id']}"
        tokenizer_str += f"\nPAD: {vocab_info['pad']} id: {vocab_info['pad_id']}"
        tokenizer_str += f"\nUNK: {vocab_info['unk']} id: {vocab_info['unk_id']}"
        tokenizer_str += f"\nBase Vocab Size: {vocab_info['vocab_size']}, Full Vocab Size: {vocab_info['full_vocab_size']}"
        prompt_format = vocab_info.get('prompt_format', None)

        if vocab_info['default_prompt_format']:
            tokenizer_str += f"\n\n(Likely incorrect) Prompt Format:\n{prompt_format if prompt_format else 'Unknown'}"
        else:
            tokenizer_str += f"\n\nPrompt Format:\n{prompt_format if prompt_format else 'Unknown'}"

        if show_special_tokens:
            if vocab_info.get('all_special_tokens', {}) == {}:
                tokenizer_str += f"\n\nNo Special Tokens Could Be Fetched"
            else:
                tokenizer_str += f"\n\nAll Special Tokens:"
                for token, id in vocab_info['all_special_tokens'].items():
                    tokenizer_str += f"\n{token} id: {id}"
        
        tokenizer_results.append(tokenizer_str)

        # Config info
        attn_impl = "Unknown"

        if isinstance(vocab_info['num_kv_heads'], int) and isinstance(vocab_info['num_heads'], int):
            if vocab_info['num_kv_heads'] == vocab_info['num_heads']:
                attn_impl = "MHA (Multi-Head Attention)"
            elif vocab_info['num_kv_heads'] == 1:
                attn_impl = "MQA (Multi-Query Attention)"
            else:
                attn_impl = "GQA (Grouped Query Attention)"

        config_str = f"{model_name} ({branch})"
        config_str += f"\nModel Type: {vocab_info['model_type']}"
        config_str += f"\nMax Context Size: {vocab_info['max_position_embeddings']}"
        config_str += f"\nArchitecture: {vocab_info['architecture']}"
        config_str += f"\nActivation Function: {vocab_info['activation_function']}"
        config_str += f"\nNumber of Layers: {vocab_info['num_layers']}"
        config_str += f"\nNumber of Attn Heads: {vocab_info['num_heads']}"
        config_str += f"\nNumber of KV Heads: {vocab_info['num_kv_heads']}"
        config_str += f"\nAttention Implementation: {attn_impl}"
        config_str += f"\nHidden Size: {vocab_info['hidden_size']}"
        config_str += f"\nIntermediate Size: {vocab_info['intermediate_size']}"
        config_str += f"\nTorch dtype: {vocab_info['torch_dtype']}"

        config_results.append(config_str)

    history_display = update_history_display()
    return ('\n' + '-' * 130 + '\n').join(tokenizer_results), ('\n' + '-' * 130 + '\n').join(config_results), history_display

def change_visibility(toggle, text):
    num_lines = len(text.split("\n")) - 1
    return gr.TextArea(visible=toggle, lines=num_lines if num_lines > 0 else 7)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=40):
            model_input = gr.Textbox(label="Model Name", show_label=False, placeholder="author/model_name", info="Accepts author/model, and full URLs to model repos, e.g. huggingface.co/author/model")
            with gr.Row():
                show_special_tokens = gr.Checkbox(label="Show all special tokens", value=False, interactive=True)
                use_local_cache = gr.Checkbox(label="Use local cache", value=True, interactive=True)
                show_model_config = gr.Checkbox(label="Show model config", value=True, interactive=True)

        submit_button = gr.Button(value="Retrieve Model Info", scale=18.5)
    with gr.Row():
        model_tokenizer_output = gr.TextArea(label="Tokenizer Information", interactive=False, autoscroll=False)
        model_config_output = gr.TextArea(label="Config Information", interactive=False, autoscroll=False, visible=show_model_config)
        with gr.Column():
            model_history_output = gr.TextArea(label="Model History", interactive=False, value=update_history_display(), info="author/model_name (branch): vocab_family")
            clear_history_button = gr.Button(value="Clear History", interactive=True)
        
    submit_button.click(
        fn=find_model_info, 
        inputs=[model_input, show_special_tokens, use_local_cache], 
        outputs=[model_tokenizer_output, model_config_output, model_history_output]
    )

    clear_history_button.click(
        fn=clear_history,
        inputs=[],
        outputs=[model_history_output]
    )

    show_model_config.change(
        fn=change_visibility,
        inputs=[show_model_config, model_config_output],
        outputs=model_config_output
    )


demo.launch(debug=True)
