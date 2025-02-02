# service/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agents import agent1_executor, agent2_executor, lc_agent1_executor, lc_agent2_executor

app = FastAPI(title="IMDB Sentiment Classifier API")

class Review(BaseModel):
    review: str
    agent_type: Optional[str] = "autogen"  # Valid values: "autogen" or "langchain"

@app.get("/")
def read_root():
    return {"message": "IMDB Sentiment Classifier API is running."}

@app.post("/predict")
def predict_sentiment(review: Review):
    if not review.review.strip():
        raise HTTPException(status_code=400, detail="Review text is empty.")
    
    if review.agent_type.lower() == "langchain":
        # Use LangChain agents
        cleaned_text = lc_agent1_executor.run(review.review)
        sentiment = lc_agent2_executor.run(cleaned_text)
    else:
        # Default: Use Autogen agents
        cleaned_text = agent1_executor.run(review.review)
        sentiment = agent2_executor.run(cleaned_text)
        
    return {"sentiment": sentiment}
