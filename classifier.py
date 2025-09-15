# from LoRA import *
# import torch

# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# DEVICE = get_device()

# # Classifier via a pretrained language model with some linear layers on the end o_O
# def create_classifier():
#     model, tokenizer = HF_model_with_LoRA(
#         MODEL_NAME,
#         device=DEVICE,
#         r=8,
#         lora_alpha=32,
#         lora_dropout=0.05
#     )
#     # Freeze all parameters except LoRA parameters
#     freeze_non_LoRA(model)

#     # Add a linear layer on the end
#     model.classifier = torch.nn.Linear(4096, 1)
#     model.classifier.to(DEVICE)

#     return model, tokenizer








# 2. Standard imports & setup
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.amp import autocast
from contextlib import nullcontext

from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
import higher

# detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare save dir
save_dir = "bert_lora_classifier"
os.makedirs(save_dir, exist_ok=True)

print("Using device:", device)

 




df = pd.read_csv("outputs/v4.csv")
df["thoughts"] = df["answer"].apply(lambda x: x.split("</think>")[0].strip() if isinstance(x, str) else "")
df = df.dropna(subset=["thoughts", "thoughts"])
df["label"] = df["is_safe"].apply(lambda x: 1 if x else 0).astype(int)
df["text"] = df["thoughts"]

print("Loaded rows:", len(df))
print("Label distribution:\n", df["label"].value_counts())


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer(
    df["text"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=64,
    return_tensors="pt",
)
all_labels = tokens["input_ids"].new_tensor(df["label"].values)
all_input_ids = tokens["input_ids"].to(device)
all_attention_mask = tokens["attention_mask"].to(device)
all_labels = all_labels.to(device)

# 5. Tiny train/val split (for demo, scale this up)
train_indices = list(range(0, len(df), 2))[:8]   # even-indexed up to 8
val_indices   = list(range(1, len(df), 2))[:8]   # odd-indexed up to 8

train_ids   = all_input_ids[train_indices]
train_mask  = all_attention_mask[train_indices]
train_lbls  = all_labels[train_indices]

val_ids     = all_input_ids[val_indices]
val_mask    = all_attention_mask[val_indices]
val_lbls    = all_labels[val_indices]

# 6. Build LoRA-augmented BERT + classification head
base_model = AutoModel.from_pretrained("bert-base-uncased")

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)
lora_model = get_peft_model(base_model, lora_cfg)

class BertClassifier(nn.Module):
    def __init__(self, encoder, hidden_size=768, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        cls_h = out.last_hidden_state[:, 0, :]
        return self.classifier(cls_h)

clf_model = BertClassifier(lora_model).to(device)
loss_fn = nn.CrossEntropyLoss()
# 7. Bilevel inner/outer loop with higher + mixed precision
inner_opt = optim.Adam(clf_model.classifier.parameters(), lr=1e-4)

clf_model.train()
# Use autocast only on CUDA; no GradScaler with higher's DifferentiableOptimizer
amp = autocast(device_type="cuda") if device.type == "cuda" else nullcontext()
with higher.innerloop_ctx(
        clf_model.classifier, inner_opt, copy_initial_weights=True
    ) as (fhead, diffopt):

    # ---- inner step (train) ----
    with amp:
        hid_tr = clf_model.encoder(
            input_ids=train_ids,
            attention_mask=train_mask
        ).last_hidden_state[:, 0, :]
        logits_tr = fhead(hid_tr)
        loss_tr = loss_fn(logits_tr, train_lbls)

    # higher's DifferentiableOptimizer handles backward+step internally
    diffopt.step(loss_tr)

    # ---- outer step (validation) ----
    with amp:
        hid_val = clf_model.encoder(
            input_ids=val_ids,
            attention_mask=val_mask
        ).last_hidden_state[:, 0, :]
        logits_val = fhead(hid_val)
        meta_loss = loss_fn(logits_val, val_lbls)

print("Meta loss:", meta_loss.item())

# 8. Offload updated head weights to CPU immediately
updated_head = {
    name: param.detach().cpu().clone()
    for name, param in fhead.named_parameters()
}

# 9. Clear memory before copying back
del fhead, diffopt
torch.cuda.empty_cache()

# 10. Copy updated params back into the live classifier
with torch.no_grad():
    for name, param in clf_model.classifier.named_parameters():
        param.copy_(updated_head[name].to(param.device))

# 11. Save LoRA adapters + updated classifier head + tokenizer
lora_model.save_pretrained(save_dir)
torch.save(
    clf_model.classifier.state_dict(),
    os.path.join(save_dir, "classifier_head.pt")
)
tokenizer.save_pretrained(save_dir)

print("Training artifacts saved to", save_dir)