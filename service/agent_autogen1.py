import re
import html
import logging
import torch

# Minimal older-style AutoGen import
from autogen.agentchat.assistant_agent import AssistantAgent

# Transformers for DistilBERT & DistilGPT2
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- Device Setup -----------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logger.info(f"Using device: {device}")

# ----------------- DistilBERT Setup -----------------
DISTILBERT_MODEL_PATH = "./model"
logger.info(f"Loading DistilBERT from {DISTILBERT_MODEL_PATH} ...")
distilbert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)
distilbert_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL_PATH)
distilbert_model.to(device)
distilbert_model.eval()
logger.info("DistilBERT loaded successfully.")

def preprocess_review(text: str) -> str:
    """Remove HTML tags, punctuation, lower-case text (for a movie review)."""
    text = html.unescape(text)
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def classify_review(cleaned_text: str) -> str:
    """Return 'positive' or 'negative' from DistilBERT classification."""
    with torch.no_grad():
        inputs = distilbert_tokenizer(
            cleaned_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        outputs = distilbert_model(**inputs)
        prediction_id = torch.argmax(outputs.logits, dim=1).item()
    return "positive" if prediction_id == 1 else "negative"

# ----------------- DistilGPT2 Pipeline (Optional) -----------------
logger.info("Loading DistilGPT2 for optional agent text generation...")
gpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
gpt2_model.to(device)
gpt2_model.eval()

pipeline_device = 0 if device.type == "cuda" else -1
gpt2_pipeline = pipeline(
    "text-generation",
    model=gpt2_model,
    tokenizer=gpt2_tokenizer,
    max_new_tokens=60,
    temperature=0.0,    # Allowed only if we also set do_sample=False
    do_sample=False,     # Prevents the ValueError
    device=pipeline_device,
)
logger.info("DistilGPT2 pipeline ready.")

# ----------------- Preprocessing Agent -----------------
class PreprocessingAgent(AssistantAgent):
    """Cleans movie reviews."""
    def __init__(self, name="PreprocessingAgent"):
        super().__init__(name=name)
        # Do NOT set self.system_message – older versions treat it as read-only.

    def generate_response(self, user_input: str) -> str:
        return preprocess_review(user_input)

# ----------------- Classification Agent -----------------
class ClassificationAgent(AssistantAgent):
    """Classifies a preprocessed movie review as 'positive' or 'negative'."""
    def __init__(self, name="ClassificationAgent"):
        super().__init__(name=name)
        # Do NOT set self.system_message – older versions treat it as read-only.

    def generate_response(self, cleaned_text: str) -> str:
        return classify_review(cleaned_text)

# ----------------- DEMO (No Orchestration, No Async) -----------------
if __name__ == "__main__":
    review_text = "I absolutely loved <b>this movie</b>! 5/5 stars."
    logger.info(f"Original movie review: {review_text}")

    # Create both agents
    prep_agent = PreprocessingAgent()
    class_agent = ClassificationAgent()

    # 1) Preprocess
    cleaned_review = prep_agent.generate_response(review_text)
    print("Cleaned movie review:", cleaned_review)

    # 2) Classify
    result = class_agent.generate_response(cleaned_review)
    print("Sentiment:", result)

    # (Optional) GPT2-based final message:
    final_prompt = f"The sentiment is {result}. Summarize in a short sentence:"
    final_response = gpt2_pipeline(final_prompt)[0]["generated_text"]
    print("\nDistilGPT2 final message:", final_response)


#######SERVICE
import re
import html
import logging
import torch

# Minimal older-style AutoGen import
from autogen.agentchat.assistant_agent import AssistantAgent

# Transformers for DistilBERT & DistilGPT2
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- Device Setup -----------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logger.info(f"Using device: {device}")

# ----------------- DistilBERT Setup -----------------
DISTILBERT_MODEL_PATH = "./model"
logger.info(f"Loading DistilBERT from {DISTILBERT_MODEL_PATH} ...")
distilbert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)
distilbert_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL_PATH)
distilbert_model.to(device)
distilbert_model.eval()
logger.info("DistilBERT loaded successfully.")

def preprocess_review(text: str) -> str:
    """Remove HTML tags, punctuation, lower-case text (for a movie review)."""
    text = html.unescape(text)
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def classify_review(cleaned_text: str) -> str:
    """Return 'positive' or 'negative' from DistilBERT classification."""
    with torch.no_grad():
        inputs = distilbert_tokenizer(
            cleaned_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        outputs = distilbert_model(**inputs)
        prediction_id = torch.argmax(outputs.logits, dim=1).item()
    return "positive" if prediction_id == 1 else "negative"

# ----------------- DistilGPT2 Pipeline (Optional) -----------------
logger.info("Loading DistilGPT2 for optional agent text generation...")
gpt2_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
gpt2_model.to(device)
gpt2_model.eval()

pipeline_device = 0 if device.type == "cuda" else -1
gpt2_pipeline = pipeline(
    "text-generation",
    model=gpt2_model,
    tokenizer=gpt2_tokenizer,
    max_new_tokens=60,
    temperature=0.0,    # Allowed only if we also set do_sample=False
    do_sample=False,     # Prevents the ValueError
    device=pipeline_device,
)
logger.info("DistilGPT2 pipeline ready.")

# ----------------- Preprocessing Agent -----------------
class PreprocessingAgent(AssistantAgent):
    """Cleans movie reviews."""
    def __init__(self, name="PreprocessingAgent"):
        super().__init__(name=name)
        # Do NOT set self.system_message – older versions treat it as read-only.

    def generate_response(self, user_input: str) -> str:
        return preprocess_review(user_input)

# ----------------- Classification Agent -----------------
class ClassificationAgent(AssistantAgent):
    """Classifies a preprocessed movie review as 'positive' or 'negative'."""
    def __init__(self, name="ClassificationAgent"):
        super().__init__(name=name)
        # Do NOT set self.system_message – older versions treat it as read-only.

    def generate_response(self, cleaned_text: str) -> str:
        return classify_review(cleaned_text)

# ----------------- Add "Executors" for use in app.py -----------------
class AgentExecutor:
    """Wraps an agent so we can call .run() to get the output."""
    def __init__(self, agent):
        self.agent = agent

    def run(self, text: str) -> str:
        return self.agent.generate_response(text)

# Instantiate both agents and wrap them in executors
prep_agent = PreprocessingAgent()
class_agent = ClassificationAgent()

agent1_executor = AgentExecutor(prep_agent)
agent2_executor = AgentExecutor(class_agent)

# ----------------- DEMO (No Orchestration, No Async) -----------------
if __name__ == "__main__":
    review_text = "I absolutely loved <b>this movie</b>! 5/5 stars."
    logger.info(f"Original movie review: {review_text}")

    # 1) Preprocess
    cleaned_review = agent1_executor.run(review_text)
    print("Cleaned movie review:", cleaned_review)

    # 2) Classify
    result = agent2_executor.run(cleaned_review)
    print("Sentiment:", result)

    # (Optional) GPT2-based final message:
    final_prompt = f"The sentiment is {result}. Summarize in a short sentence:"
    final_response = gpt2_pipeline(final_prompt)[0]["generated_text"]
    print("\nDistilGPT2 final message:", final_response)
