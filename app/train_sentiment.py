import os
import sys
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "1"
os.environ["DISABLE_WANDB_INTEGRATION"] = "1"
os.environ["DISABLE_DAGSHUB_INTEGRATION"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
import numpy as np
import mlflow
import dagshub
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from datasets import Dataset
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score, mean_absolute_error, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer)
from app.config import (MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, MODEL_FILE, METRICS_FILE, ARTIFACT_DIR, MODEL_DIR)

dagshub.init(repo_owner='ChaitanyaC20', repo_name='Sentiment_Analysis', mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT or "Sentiment")

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        labels = inputs.get("labels").float().to(device)
        outputs = model(**{k: v.to(device) for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        preds = torch.softmax(logits, dim=1)
        values = torch.arange(5, dtype=torch.float, device=device)
        expected = (preds * values).sum(1)
        l1_loss = SmoothL1Loss()(expected, labels)
        ce_loss = CrossEntropyLoss(label_smoothing=0.05)(logits, labels.long())
        loss = 0.8 * ce_loss + 0.2 * l1_loss
        return (loss, outputs) if return_outputs else loss

def train_sentiment(df: pd.DataFrame):
    model_name = "tabularisai/robust-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    dataset = Dataset.from_pandas(df[['Cleaned_Reviews', 'labels']])
    dataset = dataset.map(lambda batch: tokenizer(batch['Cleaned_Reviews'], truncation=True, padding='max_length', max_length=256), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset, eval_dataset = split['train'], split['test']
    classes = np.unique(df["Satisfaction_Score"])
    weights = compute_class_weight('balanced', classes=classes, y=df["Satisfaction_Score"])
    weights = torch.tensor(weights / weights.sum() * len(weights), dtype=torch.float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not detected — using CPU.")
    use_cuda = torch.cuda.is_available()
    model.to(device)

    training_args = TrainingArguments(
    output_dir="./artifacts_telco_sentiment",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_accumulation_steps=2,
    fp16=use_cuda,          
    torch_compile=False, 
    report_to=[])

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer)

    with mlflow.start_run(run_name="FineTune_RobustSentiment_1"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", training_args.num_train_epochs)
        mlflow.log_param("learning_rate", training_args.learning_rate)
        mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
        mlflow.log_param("device", "GPU" if use_cuda else "CPU")

        trainer.train()

        preds = trainer.predict(eval_dataset)
        pred_labels = np.argmax(preds.predictions, axis=1) + 1
        true_labels = preds.label_ids + 1

        acc = accuracy_score(true_labels, pred_labels)
        qwk = cohen_kappa_score(true_labels, pred_labels, weights='quadratic')
        mae = mean_absolute_error(true_labels, pred_labels)
        metrics = {"accuracy": acc, "quadratic_weighted_kappa": qwk, "mean_absolute_error": mae}

        print("\nEvaluation Results:")
        print(metrics)
        print("\nDetailed Report:\n", classification_report(true_labels, pred_labels, digits=3, zero_division=0))
        mlflow.log_metrics(metrics)

        print("\nGenerating evaluation visualizations")

        plt.figure(figsize=(8, 5))
        plt.hist(true_labels, bins=np.arange(1, 7) - 0.5, alpha=0.6, label="Actual", edgecolor='black')
        plt.hist(pred_labels, bins=np.arange(1, 7) - 0.5, alpha=0.6, label="Predicted", edgecolor='black')
        plt.legend()
        plt.title("Actual vs Predicted Sentiment Distribution")
        plt.xlabel("Score (1-5)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig("actual_vs_predicted_distribution.png")
        mlflow.log_artifact("actual_vs_predicted_distribution.png")
        plt.close()

        cm = confusion_matrix(true_labels, pred_labels, labels=[1, 2, 3, 4, 5])
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(7, 6))
        sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Normalized Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close('all')

        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

        model_save_path = os.path.join(MODEL_DIR, "telco_sentiment_model")
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_save_path, "model_state_dict.pt"))

        print(f"\nModel saved locally at: {MODEL_DIR}")
        print(f"Metrics logged to MLflow experiment: {MLFLOW_EXPERIMENT}")
        sys.exit(0)
    return {"metrics": metrics, "model_path": MODEL_DIR}

if __name__ == "__main__":
    from app.data import load_sentiment_data
    from app.preprocess_sentiment import preprocess_reviews
    df = load_sentiment_data()
    clean_df = preprocess_reviews(df)
    train_sentiment(clean_df)