from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode_text(text):
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )