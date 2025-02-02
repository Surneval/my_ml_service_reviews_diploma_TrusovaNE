# scripts/train.py

import os
from dotenv import load_dotenv
from comet_ml import Experiment

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

def main():
    """
    This script trains a sentiment classification model on the IMDB dataset using:
    1. A hold-out validation approach (train.csv and val.csv).
    2. LoRA (Low-Rank Adaptation) to fine-tune a DistilBERT model.
    3. Comet.ml for experiment tracking.
    4. MLflow as an additional experiment logger.

    Validation Approach:
    --------------------
    - We use a hold-out validation split, where the data is already split into
      train.csv (training data) and val.csv (validation data).
    - No k-fold cross-validation is performed in this script.
    - We assume that the provided splits are representative, and no special
      adjustments for class imbalance or time-based splitting are taken in this script.
    - If dealing with class imbalance, consider techniques like oversampling,
      undersampling, or class-weighting. These are not implemented here.
    """
    try:
        # Load environment variables from .env file
        load_dotenv()

        # Initialize Comet.ml Experiment
        COMET_API_KEY = os.getenv("COMET_API_KEY")
        COMET_WORKSPACE = os.getenv("COMET_WORKSPACE")
        COMET_PROJECT_NAME = os.getenv("COMET_PROJECT_NAME")
        MODEL_DIR = os.getenv("MODEL_DIR")

        if not COMET_API_KEY:
            raise ValueError("COMET_API_KEY is not set in the environment variables.")
        if not MODEL_DIR:
            raise ValueError("MODEL_DIR is not set in the environment variables.")

        experiment = Experiment(
            api_key=COMET_API_KEY,
            workspace=COMET_WORKSPACE,
            project_name=COMET_PROJECT_NAME,
            log_code=True,  # Automatically log the code
            auto_output_logging="simple"  # Automatically log stdout and stderr
        )

        # Set experiment name (optional)
        experiment.set_name("IMDB_LoRA_Classification")

        # Logging configuration
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info("Loading data...")
        # The dataset is split beforehand into training and validation
        # Hence, we are using a hold-out validation approach here.
        train_df = pd.read_csv("data/train.csv")
        val_df = pd.read_csv("data/val.csv")

        model_name = "distilbert-base-uncased"
        logger.info(f"Loading tokenizer and model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # Log model configuration to Comet.ml
        experiment.log_parameter("model_name", model_name)

        # LoRA Configuration with correct target_modules
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

        # Log LoRA configuration to Comet.ml
        experiment.log_parameters({
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "bias": lora_config.bias,
            "task_type": lora_config.task_type,
            "target_modules": lora_config.target_modules
        })

        # Create datasets
        logger.info("Creating datasets...")
        train_dataset = IMDBDataset(
            texts=train_df['clean_review'].tolist(),
            labels=train_df['label'].tolist(),
            tokenizer=tokenizer
        )
        val_dataset = IMDBDataset(
            texts=val_df['clean_review'].tolist(),
            labels=val_df['label'].tolist(),
            tokenizer=tokenizer
        )

        # Determine device
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        model.to(device)
        logger.info(f"Using device: {device}")
        experiment.log_parameter("device", str(device))

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="outputs",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="logs",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to=["comet_ml"],  # Enable Comet.ml reporting
            run_name="IMDB_LoRA_Training"  # Name for the run in Comet.ml
        )

        # Log training arguments to Comet.ml
        experiment.log_parameters({
            "output_dir": training_args.output_dir,
            "eval_strategy": training_args.eval_strategy,
            "save_strategy": training_args.save_strategy,
            "logging_dir": training_args.logging_dir,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
            "num_train_epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "weight_decay": training_args.weight_decay,
            "logging_steps": training_args.logging_steps,
            "load_best_model_at_end": training_args.load_best_model_at_end,
            "metric_for_best_model": training_args.metric_for_best_model,
            "report_to": training_args.report_to,
            "run_name": training_args.run_name
        })

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = logits.argmax(axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        # MLflow Configuration
        mlflow_tracking_uri = "file://" + os.path.abspath("mlruns")
        mlflow_experiment_name = "imdb-lora-classification"
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
        logger.info(f"Setting MLflow tracking URI to {mlflow_tracking_uri}")
        logger.info(f"Setting MLflow experiment name to {mlflow_experiment_name}")

        # Log MLflow configurations to Comet.ml
        experiment.log_parameters({
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_experiment_name": mlflow_experiment_name
        })

        # Initialize Trainer with MLflowCallback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=1),
                MLflowCallback()
            ]
        )

        logger.info("Starting training...")
        experiment.log_other("Training Status", "Started")
        trainer.train()
        metrics = trainer.evaluate(val_dataset)
        logger.info(f"Evaluation metrics: {metrics}")

        # Log evaluation metrics to Comet.ml
        experiment.log_metrics(metrics)

        # Save model and tokenizer
        logger.info("Saving model and tokenizer...")
        trainer.save_model(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        logger.info(f"Model and tokenizer saved to '{MODEL_DIR}' directory.")

        # Verify the directory contents
        if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
            logger.info(f"'{MODEL_DIR}' directory contents: {os.listdir(MODEL_DIR)}")
            # Log the artifact to Comet.ml
            experiment.log_artifact(MODEL_DIR, artifact_file='model', overwrite=True)
        else:
            logger.error(f"'{MODEL_DIR}' directory is empty after saving.")
            experiment.log_other("Model Save Status", f"'{MODEL_DIR}' directory is empty after saving.")

        logger.info("Model and tokenizer saved to 'model/' directory.")

        # Optionally, you can log additional artifacts or information here

        experiment.log_other("Training Status", "Completed")

    except ValueError as ve:
        logger = logging.getLogger(__name__)
        logger.error(f"ValueError: {ve}")
        if 'experiment' in locals():
            experiment.log_other("Training Status", f"Failed with ValueError: {ve}")
            experiment.end()
        raise
    except FileNotFoundError as fe:
        logger = logging.getLogger(__name__)
        logger.error(f"FileNotFoundError: {fe}")
        if 'experiment' in locals():
            experiment.log_other("Training Status", f"Failed with FileNotFoundError: {fe}")
            experiment.end()
        raise
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"An unexpected error occurred: {e}")
        if 'experiment' in locals():
            experiment.log_other("Training Status", f"Failed with Exception: {e}")
            experiment.end()
        raise
    finally:
        try:
            if 'experiment' in locals():
                experiment.end()
        except:
            pass

if __name__ == "__main__":
    main()
