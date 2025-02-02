import os
from dotenv import load_dotenv
import pandas as pd
import torch
import logging
import numpy as np
import evaluate  # Make sure to install this: pip install evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from utils import IMDBDataset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ----------------------------
# Environment and Configuration
# ----------------------------

# Load environment variables from .env file (if available)
load_dotenv()

# Use the value from the environment variable MODEL_DIR if set; otherwise, fall back to a default path.
MODEL_DIR = os.getenv("MODEL_DIR", "/Users/Nadia/Downloads/my_ml_service_reviews_diploma/model")
print(f"Using MODEL_DIR: {MODEL_DIR}")

# You can set the model name as needed.
model_name = "distilbert-base-uncased"

# ----------------------------
# Set up Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Load Tokenizer and Fine-Tuned Model
# ----------------------------

logger.info(f"Loading tokenizer from {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

logger.info(f"Loading fine-tuned model from {MODEL_DIR}")
# If your model was fine-tuned using LoRA/PEFT, ensure that the proper configuration is used.
# Uncomment and adjust the following lines if you need to rebuild the PEFT model:
#
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.1,
#     bias="none",
#     task_type="SEQ_CLS",
#     target_modules=["q_lin", "v_lin"]
# )
# base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
# model = get_peft_model(base_model, lora_config)
#
# For the purpose of this script, we assume that MODEL_DIR contains the final fine-tuned model.
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----------------------------
# Load Validation Data
# ----------------------------

logger.info("Loading validation data...")
val_df = pd.read_csv("data/val.csv")
# Create the validation dataset from the DataFrame
val_dataset = IMDBDataset(
    texts=val_df['clean_review'].tolist(),
    labels=val_df['label'].tolist(),
    tokenizer=tokenizer
)

# ----------------------------
# Prepare the Trainer (for prediction)
# ----------------------------

# Minimal training arguments are needed just for running predictions.
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_eval_batch_size=16
)

# ----------------------------
# Load Evaluation Metrics
# ----------------------------

# Load evaluation metrics
metric_accuracy = evaluate.load("accuracy")
metric_precision = evaluate.load("precision")
metric_recall = evaluate.load("recall")
metric_f1 = evaluate.load("f1")

# ----------------------------
# Define the compute_metrics Function
# ----------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Compute each metric
    acc = metric_accuracy.compute(predictions=predictions, references=labels)
    prec = metric_precision.compute(predictions=predictions, references=labels, average='binary')
    rec = metric_recall.compute(predictions=predictions, references=labels, average='binary')
    f1 = metric_f1.compute(predictions=predictions, references=labels, average='binary')
    
    # Combine all metrics into a single dictionary
    return {
        "accuracy": acc["accuracy"],
        "precision": prec["precision"],
        "recall": rec["recall"],
        "f1": f1["f1"]
    }

# Initialize the Trainer with the updated compute_metrics
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ----------------------------
# Get Predictions on the Validation Set
# ----------------------------

logger.info("Computing predictions on the validation dataset...")
prediction_output = trainer.predict(val_dataset)
logits = prediction_output.predictions
true_labels = prediction_output.label_ids

# Compute predicted labels (assumes binary classification using argmax)
predicted_labels = np.array(logits.argmax(axis=-1))
true_labels = np.array(true_labels)

# ----------------------------
# Error Analysis: FP/FN, Confusion Matrix, and Classification Report
# ----------------------------

# For binary classification, assume:
#   - Class 0: Negative
#   - Class 1: Positive
# (Adjust as needed for your labeling scheme.)

# Retrieve all review texts from the validation DataFrame.
val_texts = val_df['clean_review'].tolist()

false_positives = []  # Prediction: 1 (Positive) but True Label: 0 (Negative)
false_negatives = []  # Prediction: 0 (Negative) but True Label: 1 (Positive)

for idx, (pred, true) in enumerate(zip(predicted_labels, true_labels)):
    if pred != true:
        if pred == 1 and true == 0:
            false_positives.append({
                "index": idx,
                "text": val_texts[idx],
                "predicted": pred,
                "true": true
            })
        elif pred == 0 and true == 1:
            false_negatives.append({
                "index": idx,
                "text": val_texts[idx],
                "predicted": pred,
                "true": true
            })

logger.info(f"Total false positives: {len(false_positives)}")
logger.info(f"Total false negatives: {len(false_negatives)}")

# Calculate and log confusion matrix and classification report
cm = confusion_matrix(true_labels, predicted_labels)
logger.info("Confusion Matrix:")
logger.info(cm)

report = classification_report(true_labels, predicted_labels, target_names=["Negative", "Positive"])
logger.info("Classification Report:")
logger.info(report)

# Optionally, save the confusion matrix and classification report as text files.
with open("confusion_matrix.txt", "w") as cm_file:
    cm_file.write(str(cm))
with open("classification_report.txt", "w") as cr_file:
    cr_file.write(report)

# ----------------------------
# Save FP and FN Analysis to CSV Files
# ----------------------------

df_fp = pd.DataFrame(false_positives)
df_fn = pd.DataFrame(false_negatives)

fp_csv_filename = "all_false_positives.csv"
fn_csv_filename = "all_false_negatives.csv"

df_fp.to_csv(fp_csv_filename, index=False)
df_fn.to_csv(fn_csv_filename, index=False)

logger.info(f"False positives saved to {fp_csv_filename}")
logger.info(f"False negatives saved to {fn_csv_filename}")

# ----------------------------
# Log and Save Additional Metrics
# ----------------------------

# Extract metrics from prediction_output
metrics = prediction_output.metrics
accuracy = metrics.get("test_accuracy")
precision = metrics.get("test_precision")
recall = metrics.get("test_recall")
f1 = metrics.get("test_f1")

# Log the metrics
logger.info(f"Accuracy: {accuracy:.4f}")
logger.info(f"Precision: {precision:.4f}")
logger.info(f"Recall: {recall:.4f}")
logger.info(f"F1 Score: {f1:.4f}")

# Optionally, save these metrics to a text file
with open("evaluation_metrics.txt", "w") as em_file:
    em_file.write(f"Accuracy: {accuracy:.4f}\n")
    em_file.write(f"Precision: {precision:.4f}\n")
    em_file.write(f"Recall: {recall:.4f}\n")
    em_file.write(f"F1 Score: {f1:.4f}\n")

logger.info("Evaluation metrics saved to evaluation_metrics.txt")

# ----------------------------
# ROC Curve Analysis
# ----------------------------

logger.info("Starting ROC curve analysis...")

# Convert logits to probabilities for the positive class
probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
roc_auc = auc(fpr, tpr)

logger.info(f"ROC AUC: {roc_auc:.4f}")

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()

logger.info("ROC curve saved as roc_curve.png")
