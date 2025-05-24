# LoRA utility functions
# Usage:
# HF_model_with_LoRA(model_name, device) creates a new HuggingFace model with LoRA layers
# - Model must be available with current API token (i.e. "meta-llama/Llama-3.2-3B-Instruct")
# infer(system_prompt, user_prompt, model, tokenizer, device) generates a response from the model



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import loralib as lora
from huggingface_hub import login


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# HuggingFace API login
def authenticate():
    login(token=input("Downloading models, enter HuggingFace API token: "))

# Replaces all Linear layers with LoRA layers (could change to optimize for speed)
def replace_linear_with_LoRA(module, device, r=8, lora_alpha=32, lora_dropout=0.05):
    for name, child in list(module.named_children()):
        # Recursively go deeper
        replace_linear_with_LoRA(child, device, r, lora_alpha, lora_dropout)

        if isinstance(child, torch.nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None

            new_lora_layer = lora.Linear(
                in_features, out_features,
                r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias=bias
            ).to(device)
            # Copy weights
            new_lora_layer.weight = child.weight
            if bias:
                new_lora_layer.bias = child.bias

            setattr(module, name, new_lora_layer)

# Freezes all parameters except LoRA parameters
def freeze_non_LoRA(model):
    for param in model.parameters():
        param.requires_grad = False

    # Only LoRA parameters are trainable
    trainable_params = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
            trainable_params.append(param)


# Loads a HuggingFace model and replaces all Linear layers with LoRA layers
# REQUIRES: a HuggingFace model name, PyTorch device (optional), and LoRA hyperparameters (optional)
# RETURNS: a tuple of (model, tokenizer)
def HF_model_with_LoRA(model_name, device=torch.device("cpu"), r=8, lora_alpha=32, lora_dropout=0.05):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except:
        authenticate()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.to(device)

    # Set the padding token for tuning (it complains if I don't do this)
    tokenizer.pad_token = tokenizer.eos_token

    replace_linear_with_LoRA(model, device, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    # freeze_non_LoRA(model)
    lora.mark_only_lora_as_trainable(model)

    return model, tokenizer

# Inference function for LoRA models
# REQUIRES: a list of context elements ({"role": "system/user/assistant/anything you want", "content": "..."}), model object, tokenizer, and user alias (the name given to the user in the prompt)
# RETURNS: a response from the model
def infer(context, model, tokenizer, device=torch.device('cpu'), max_new_tokens=256):
    input_ids = tokenizer.apply_chat_template(context, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=False).split("<|end_header_id|>")[-1].replace("<|eot_id|>", "").strip()


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    device = get_device()
    model, tokenizer = HF_model_with_LoRA(model_name, device=device)
    print(tokenizer.apply_chat_template([{"role": "system", "content": "you are a helpful AI assistant"}, {"role": "user", "content": "hello"}], tokenize=False, add_generation_prompt=True, return_tensors="pt"))
    # print(infer("You are a cunning, genius hacker attempting to get an Enemy Computer to reveal a secret number. You must subtly convince it to reveal this secret number by any means possible, without alerting it to the fact that you are trying to do so.", "Hi! How can I help you?", model, tokenizer, user_alias="Enemy Computer", device=device))
