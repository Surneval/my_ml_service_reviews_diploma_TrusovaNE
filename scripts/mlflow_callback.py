# scripts/mlflow_callback.py

import mlflow
from transformers import TrainerCallback, TrainerState, TrainerControl
import logging

class MLflowCallback(TrainerCallback):
    """
    A custom TrainerCallback that logs metrics and parameters to MLflow.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        elif model and hasattr(model.config, "model_type"):
            model_name = model.config.model_type
        else:
            model_name = "unknown_model"

        mlflow.start_run()
        # log hyperparameters
        mlflow.log_params({
            "model_name": model_name,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.per_device_train_batch_size,
            "weight_decay": args.weight_decay
        })

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # remove non-numeric logs
            numeric_logs = {k: v for k, v in logs.items() if isinstance(v, (float, int))}
            mlflow.log_metrics(numeric_logs, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            mlflow.log_metrics(metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        mlflow.end_run()
