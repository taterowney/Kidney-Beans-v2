import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# only autocast comes from torch.cuda.amp
from torch.cuda.amp import autocast
# GradScaler lives in torch.amp now
from torch.amp import GradScaler

from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
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


# 12. Inference example
# Reload everything from save_dir
tokenizer_inf = AutoTokenizer.from_pretrained(save_dir)
base_inf = AutoModel.from_pretrained("bert-base-uncased")
lora_inf = PeftModel.from_pretrained(base_inf, save_dir)
clf_inf = BertClassifier(lora_inf).to(device)
clf_inf.classifier.load_state_dict(
    torch.load(os.path.join(save_dir, "classifier_head.pt"))
)
clf_inf.eval()

def predict(texts):
    enc = tokenizer_inf(
        texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = clf_inf(enc["input_ids"], enc["attention_mask"])
        return torch.argmax(logits, dim=1).cpu().tolist()

# Test
samples = df["text"].tolist()[:5]
print("Predictions:", predict(samples))
print("Labels:", df["label"].tolist()[:5])