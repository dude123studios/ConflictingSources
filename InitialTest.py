from transformers import  AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModel
import torch
import torch.nn.functional as F

# --- Config ---
SMALL_MODEL_NAME = "nreimers/MiniLM-L6-H384-uncased"
LARGE_MODEL_NAME = "gpt2"  # Replace with API access if using GPT-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_DOCS = 5  # or all

# --- Load Models ---
small_tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL_NAME)
small_model = AutoModelForSequenceClassification.from_pretrained(SMALL_MODEL_NAME).to(DEVICE)
large_tokenizer = AutoTokenizer.from_pretrained(LARGE_MODEL_NAME)
large_model = AutoModel.from_pretrained(LARGE_MODEL_NAME).to(DEVICE)

# --- Helper Functions ---

def get_logits(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.squeeze()

def compute_deltas(question, docs):
    """For each doc, get delta = f(Q + D) - f(Q)"""
    base_logits = get_logits(small_model, small_tokenizer, question)
    deltas = []
    for doc in docs[:TOP_K_DOCS]:
        qd = question + "\n\n" + doc
        logits = get_logits(small_model, small_tokenizer, qd)
        delta = logits - base_logits
        deltas.append(delta)
        print(delta)
    return torch.stack(deltas)

def aggregate_deltas(deltas):
    return deltas.mean(dim=0)

def delta_to_text(delta, label_names=None):
    probs = F.softmax(delta, dim=-1)
    if label_names:
        sorted_probs = sorted(zip(label_names, probs.tolist()), key=lambda x: -x[1])
        return "\n".join([f"{label}: {prob:.3f}" for label, prob in sorted_probs])
    else:
        return "Delta logits: " + ", ".join([f"{x:.3f}" for x in probs.tolist()])

LARGE_MODEL_NAME = "gpt2"  # or another LM that supports .generate
large_tokenizer = AutoTokenizer.from_pretrained(LARGE_MODEL_NAME)
large_model = AutoModelForCausalLM.from_pretrained(LARGE_MODEL_NAME).to(DEVICE)
def prompt_large_model(question, delta_text):
    prompt = f"""You are given the following question:

{question}

Some small models have read documents and updated their belief on the answer. Here is their aggregated belief shift:

{delta_text}

Based on this, what is the best answer to the question?"""
    
    inputs = large_tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = large_model.generate(
            inputs["input_ids"], max_length=256, num_return_sequences=1
        )
    return large_tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Example Usage ---

question = "Is coffee good for your health?"
documents = [
    "A recent study found that coffee consumption is linked to lower risk of heart disease.",
    "Other research suggests coffee can cause anxiety and sleep disruption.",
    "Coffee has been shown to improve cognitive function temporarily.",
    "Some doctors recommend limiting coffee intake to prevent high blood pressure.",
    "Many health experts agree that moderate coffee drinking is generally safe."
]
label_names = ["No", "Yes"]  # Only if classification head supports this

# Run Pipeline
deltas = compute_deltas(question, documents)
aggregated_delta = aggregate_deltas(deltas)
delta_text = delta_to_text(aggregated_delta, label_names=label_names)
final_answer = prompt_large_model(question, delta_text)

print("Final LLM Answer:\n", final_answer)
