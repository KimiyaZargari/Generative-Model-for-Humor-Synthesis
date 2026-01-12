import torch
from transformers import BertTokenizer, BertForSequenceClassification

# params
model_name = 'bert-base-uncased'
model_weights_path = '/data/tarekjoshua/repos/Generative-Model-for-Humor-Synthesis/models/bert-classification-model/runs/bert_humor/20260111-232513/BERT_Humor_200k.pth'

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load tokenizer + model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# load fine-tuned weights
state_dict = torch.load(model_weights_path, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()


# inference function #

def predict_logits(text: str):
    # tokenize input text
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    # move to gpu/cpu
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # return logits for both classes
    return logits.squeeze(0).detach().cpu()


# example #

text = "I found weed in the 3rd grade': Jacob Balshin | New Wave Of Standup"
logits = predict_logits(text)
probs = torch.softmax(logits, dim=0)
print(probs)

print("Input Text:", text)
print("Logits:", probs.tolist())
print("Predicted Class:", int(torch.argmax(probs).item()))
