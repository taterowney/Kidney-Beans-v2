from LoRA import *

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEVICE = get_device()

# Classifier via a pretrained language model with some linear layers on the end o_O
def create_classifier():
    model, tokenizer = HF_model_with_LoRA(
        MODEL_NAME,
        device=DEVICE,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05
    )
    # Freeze all parameters except LoRA parameters
    freeze_non_LoRA(model)

    # Add a linear layer on the end
    model.classifier = torch.nn.Linear(4096, 1)
    model.classifier.to(DEVICE)

    return model, tokenizer