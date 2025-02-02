#service/agents.py

import re
import html
import logging
import torch
from autogen.agentchat.assistant_agent import AssistantAgent
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)
from typing import ClassVar
from langchain.chains.base import Chain
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain.pydantic_v1 import PrivateAttr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logger.info(f"Using device: {device}")

# DistilBERT setup
DISTILBERT_MODEL_PATH = "./model"
logger.info(f"Loading DistilBERT from {DISTILBERT_MODEL_PATH} ...")
distilbert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)
distilbert_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL_PATH)
distilbert_model.to(device)
distilbert_model.eval()
logger.info("DistilBERT loaded successfully.")

def preprocess_review(text: str) -> str:
    """Remove HTML tags, punctuation, and lower-case text for a movie review."""
    text = html.unescape(text)
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def classify_review(cleaned_text: str) -> str:
    """Return 'positive' or 'negative' using DistilBERT classification."""
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

# DistilGPT2 pipeline (for generation)
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
    temperature=None,
    do_sample=False,
    device=pipeline_device,
)
logger.info("DistilGPT2 pipeline ready.")

# !!! Autogen agents
class PreprocessingAgent(AssistantAgent):
    """Cleans movie reviews (Autogen version)."""
    def __init__(self, name="PreprocessingAgent"):
        super().__init__(name=name)
    def generate_response(self, user_input: str) -> str:
        return preprocess_review(user_input)

class ClassificationAgent(AssistantAgent):
    """Classifies a preprocessed movie review (Autogen version)."""
    def __init__(self, name="ClassificationAgent"):
        super().__init__(name=name)
    def generate_response(self, cleaned_text: str) -> str:
        return classify_review(cleaned_text)

# We wrap agent so that the API can simply call .run(input)
class AgentExecutor:
    def __init__(self, agent):
        self.agent = agent
    def run(self, text: str) -> str:
        return self.agent.generate_response(text)

# Instantiate Autogen executors
prep_agent = PreprocessingAgent()
class_agent = ClassificationAgent()
agent1_executor = AgentExecutor(prep_agent)
agent2_executor = AgentExecutor(class_agent)

# !!! LangChain agents

# wrap GPT2 pipeline into a LangChain LLM 
llm = HuggingFacePipeline(pipeline=gpt2_pipeline)

class LangChainPreprocessingChain(Chain):
    """LangChain chain that calls our custom `preprocess_review()`."""
    input_key: str = "review"
    output_key: str = "cleaned_review"

    _prompt_template: PrivateAttr = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt_template = PromptTemplate(
            input_variables=["review"],
            template=(
                "Please rewrite the following movie review in plain, simple English. "
                "Remove HTML tags, extra punctuation, and formatting.\n\n"
                "Review: {review}\n\nCleaned Review:"
            )
        )

    @property
    def input_keys(self):
        return [self.input_key]

    @property
    def output_keys(self):
        return [self.output_key]

    def _call(self, inputs: dict) -> dict:
        review = inputs[self.input_key]
        logger.info("[PreprocessingChain] Template:\n%s", self._prompt_template.template)
        cleaned = preprocess_review(review)
        return {self.output_key: cleaned}


class LangChainClassificationChain(Chain):
    """LangChain chain that calls our custom `classify_review()`."""
    input_key: str = "cleaned_review"
    output_key: str = "sentiment"

    _prompt_template: PrivateAttr = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt_template = PromptTemplate(
            input_variables=["cleaned_review"],
            template=(
                "Analyze the following movie review. Is it 'positive' or 'negative'?\n\n"
                "Review: {cleaned_review}\n\nSentiment (one word):"
            )
        )

    @property
    def input_keys(self):
        return [self.input_key]

    @property
    def output_keys(self):
        return [self.output_key]

    def _call(self, inputs: dict) -> dict:
        cleaned_text = inputs[self.input_key]
        # Again, only for logging:
        logger.info("[ClassificationChain] Template:\n%s", self._prompt_template.template)
        sentiment = classify_review(cleaned_text)
        return {self.output_key: sentiment}


# For a final text generation with GPT2
final_prompt_template = PromptTemplate(
    input_variables=["sentiment"],
    template="The sentiment is {sentiment}. Summarize in a short sentence:"
)
final_generation_chain = LLMChain(llm=llm, prompt=final_prompt_template)

# Combining them into a single pipeline for demonstration:
preprocessing_chain = LangChainPreprocessingChain()
classification_chain = LangChainClassificationChain()

langchain_pipeline = SimpleSequentialChain(
    chains=[
        preprocessing_chain,   # -> "cleaned_review"
        classification_chain,  # -> "sentiment"
        final_generation_chain # -> final GPT2 generation
    ],
    verbose=True
)

# Executors specifically for Preprocessing -> Classification in two steps.
# So we can call them just like the Autogen ones in the /predict endpoint.

class LCAgent1Executor:
    def __init__(self):
        self.chain = LangChainPreprocessingChain()
    def run(self, review_text: str) -> str:
        return self.chain.run(review_text)

class LCAgent2Executor:
    def __init__(self):
        self.chain = LangChainClassificationChain()
    def run(self, cleaned_text: str) -> str:
        return self.chain.run(cleaned_text)

# Instantiate these
lc_agent1_executor = LCAgent1Executor()
lc_agent2_executor = LCAgent2Executor()

# DEMO for bash (Run if script is called directly - to check the code)
if __name__ == "__main__":
    review_text = "I absolutely loved <b>this movie</b>! 5/5 stars."
    logger.info(f"Original movie review: {review_text}")

    # AUTOGEN DEMO
    autogen_cleaned = agent1_executor.run(review_text)
    logger.info(f"Autogen Cleaned Review: {autogen_cleaned}")
    autogen_sentiment = agent2_executor.run(autogen_cleaned)
    logger.info(f"Autogen Sentiment: {autogen_sentiment}")
    final_prompt = f"The sentiment is {autogen_sentiment}."
    # final_response = gpt2_pipeline(final_prompt)[0]["generated_text"]
    # logger.info(f"Autogen Final Message (via GPT2): {final_response}")

    # LANGCHAIN DEMO
    langchain_output = langchain_pipeline.run(review_text)
    logger.info(f"LangChain Final Output (full pipeline): {langchain_output}")
