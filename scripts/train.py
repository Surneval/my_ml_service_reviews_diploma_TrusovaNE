# scripts/train.py

import os
import pandas as pd
import torch
import logging
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from utils import IMDBDataset
from mlflow_callback import MLflowCallback

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Loading data...")
        train_df = pd.read_csv("data/train.csv")
        val_df = pd.read_csv("data/val.csv")

        model_name = "distilbert-base-uncased"
        logger.info(f"Loading tokenizer and model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # LoRA with correct target_modules
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["q_lin", "v_lin"]
        )
        model = get_peft_model(base_model, lora_config)

        # datasets
        logger.info("Creating datasets...")
        train_dataset = IMDBDataset(train_df['clean_review'].tolist(), train_df['label'].tolist(), tokenizer)
        val_dataset = IMDBDataset(val_df['clean_review'].tolist(), val_df['label'].tolist(), tokenizer)

        # GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        logger.info(f"Using device: {device}")

        # training args
        training_args = TrainingArguments(
            output_dir="outputs",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="logs",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="none" 
        )

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = logits.argmax(axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        # MLflow
        mlflow_tracking_uri = "file://" + os.path.abspath("mlruns")
        mlflow_experiment_name = "imdb-lora-classification"
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
        logger.info(f"Setting MLflow tracking URI to {mlflow_tracking_uri}")
        logger.info(f"Setting MLflow experiment name to {mlflow_experiment_name}")

        # Trainer with MLflowCallback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1), MLflowCallback()]
        )

        logger.info("Starting training...")
        trainer.train()
        metrics = trainer.evaluate(val_dataset)
        logger.info(f"Evaluation metrics: {metrics}")

        # save model and tokenizer
        logger.info("Saving model and tokenizer...")
        trainer.save_model("model")
        tokenizer.save_pretrained("model")
        logger.info("Model and tokenizer saved to 'model/' directory.")

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise
    except FileNotFoundError as fe:
        logger.error(f"FileNotFoundError: {fe}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main()





    # # Use GPU if available
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")  # Metal Performance Shaders on Mac M1/M2
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")  # CUDA for Nvidia GPUs
    # else:
    #     device = torch.device("cpu")  # Fallback to CPU

    # print(f"Using device: {device}")