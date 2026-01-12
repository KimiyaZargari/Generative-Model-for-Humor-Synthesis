import pandas as pd
import torch
import numpy as np
import os
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW as TorchAdamW
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# params
batch_size = 1024
lr = 3e-5
num_epochs = 3

# dataloader params
num_workers = min(4, os.cpu_count() or 1)

# training params
grad_clip_max_norm = 1.0

# fine-tuning params: fine-tune only class. head or also attn layers
fine_tune_head_only = False
unfreeze_last_n_layers = 2

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = True if device.type == "cuda" else False

df = pd.read_csv('/home/tarekjoshua/data/repos/Generative-Model-for-Humor-Synthesis/data/classification-data/classification-dataset.csv')

print(df.head())

# prepare data: convert humor column to int
df['score'] = df.pop('humor').astype(int)

print(df.head())

# get texts and labels
texts = df['text'].to_list()
labels = df['score'].to_list()

# 80, 10, 10 splits
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=1234
)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1, random_state=1234
)

# tokenizer from pre-trained BERT model and num_labels = 2 (bin. classification)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# tokenize input texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# convert to tensors (tokens, mask, label)
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids'], dtype=torch.long),
    torch.tensor(train_encodings['attention_mask'], dtype=torch.long),
    torch.tensor(train_labels, dtype=torch.long)
)
val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(val_encodings['input_ids'], dtype=torch.long),
    torch.tensor(val_encodings['attention_mask'], dtype=torch.long),
    torch.tensor(val_labels, dtype=torch.long)
)
test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(test_encodings['input_ids'], dtype=torch.long),
    torch.tensor(test_encodings['attention_mask'], dtype=torch.long),
    torch.tensor(test_labels, dtype=torch.long)
)

# dataloader (batching, shuffling)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)

# only train classification head
for param in model.bert.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

# unfreeze last n transformer layers
if not fine_tune_head_only:
    n = max(0, int(unfreeze_last_n_layers))

    # unfreeze pooler
    if hasattr(model.bert, "pooler") and model.bert.pooler is not None:
        for param in model.bert.pooler.parameters():
            param.requires_grad = True

    # unfreeze last n encoder layers
    if n > 0:
        encoder_layers = model.bert.encoder.layer
        for layer in encoder_layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True


# fine-tune BERT model #

# tensorboard writer
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=f"/home/tarekjoshua/data/repos/Generative-Model-for-Humor-Synthesis/models/bert-classification-model/runs/bert_humor/{run_name}")

# optimizer
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = TorchAdamW(trainable_params, lr=lr)

# lr scheduler (cosine annealing)
total_steps = len(train_loader) * num_epochs
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

model.train()

global_step = 0

# training loop
for epoch in range(num_epochs):

    model.train()

    for batch_idx, batch in enumerate(train_loader):

        input_ids, attention_mask, labels = batch

        # move to gpu/cpu
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_max_norm)

        optimizer.step()
        scheduler.step()

        # print loss for each batch
        print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.6f}")

        # tensorboard logging
        writer.add_scalar("train/loss", loss.item(), global_step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        global_step += 1


    # validate model #

    # eval mode
    model.eval()

    val_predictions = []
    val_true_labels = []

    val_loss_sum = 0.0
    val_batches = 0

    # val loop
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch

            # move to gpu/cpu
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            val_loss_sum += outputs.loss.item()
            val_batches += 1

            val_predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
            val_true_labels.extend(labels.detach().cpu().tolist())

    # val accuracy
    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    val_loss = val_loss_sum / max(val_batches, 1)

    print(f"Epoch {epoch+1}/{num_epochs} | Validation Loss: {val_loss:.6f} | Validation Accuracy: {val_accuracy:.6f}")

    # tensorboard logging
    writer.add_scalar("val/loss", val_loss, epoch)
    writer.add_scalar("val/accuracy", val_accuracy, epoch)


# evaluate model #

# eval mode
model.eval()

test_predictions = []
test_true_labels = []

test_loss_sum = 0.0
test_batches = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch

        # move to gpu/cpu
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        test_loss_sum += outputs.loss.item()
        test_batches += 1

        test_predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        test_true_labels.extend(labels.detach().cpu().tolist())

# test accuracy
test_accuracy = accuracy_score(test_true_labels, test_predictions)
test_loss = test_loss_sum / max(test_batches, 1)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# tensorboard logging
writer.add_scalar("test/loss", test_loss, 0)
writer.add_scalar("test/accuracy", test_accuracy, 0)
writer.close()

# save model
torch.save(model.state_dict(), f'/home/tarekjoshua/data/repos/Generative-Model-for-Humor-Synthesis/models/bert-classification-model/runs/bert_humor/{run_name}/BERT_Humor_200k.pth')
