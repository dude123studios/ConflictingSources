from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# --- Config ---
SMALL_MODEL_NAME = "nreimers/MiniLM-L6-H384-uncased"
LARGE_MODEL_NAME = "roberta-large-mnli"  # Example of open LLM that outputs logits
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Models ---
small_tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_NAME)
small_model = AutoModelForSequenceClassification.from_pretrained(SMALL_MODEL_NAME).to(DEVICE)

large_tokenizer = AutoTokenizer.from_pretrained(LARGE_MODEL_NAME)
large_model = AutoModelForSequenceClassification.from_pretrained(LARGE_MODEL_NAME).to(DEVICE)

# --- Core Functions ---

def get_logits(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.squeeze()  # shape: (num_classes)

def compute_deltas(question, docs):
    base_logits = get_logits(small_model, small_tokenizer, question)
    deltas = []
    for doc in docs:
        input_text = question + "\n\n" + doc
        logits_with_doc = get_logits(small_model, small_tokenizer, input_text)
        delta = logits_with_doc - base_logits
        deltas.append(delta)
    return torch.stack(deltas)

def aggregate_deltas(deltas):
    return deltas.mean(dim=0)

def fused_logits(question, aggregated_delta):
    base_logits_large = get_logits(large_model, large_tokenizer, question)
    final_logits = base_logits_large + aggregated_delta
    return final_logits

def predict_answer(logits, label_names):
    probs = F.softmax(logits, dim=-1)
    pred_idx = torch.argmax(probs).item()
    return label_names[pred_idx], probs.tolist()

# --- Example Run ---

question = "Is coffee good for your health?"
documents = [
    "A recent study found that coffee consumption is linked to lower risk of heart disease.",
    "Other research suggests coffee can cause anxiety and sleep disruption.",
    "Coffee has been shown to improve cognitive function temporarily.",
    "Some doctors recommend limiting coffee intake to prevent high blood pressure.",
    "Many health experts agree that moderate coffee drinking is generally safe."
]

label_names = ["No", "Yes"]  # Adjust if model has more classes

deltas = compute_deltas(question, documents)
aggregated_delta = aggregate_deltas(deltas)
final_logits = fused_logits(question, aggregated_delta)

answer, probs = predict_answer(final_logits, label_names)

print(f"Predicted Answer: {answer}")
print("Probabilities:", {k: f"{v:.3f}" for k, v in zip(label_names, probs)})
