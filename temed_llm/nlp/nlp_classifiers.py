"""
Added dependicies for the project.

!pip install -U sentence-transformers
!pip install pandas
!pip install -U matplotlib
!pip install -U seaborn
!pip install -U scikit-learn
!pip install transformers torch datasets
!pip install setfit
"""

import torch
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import accuracy_score, auc, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


class BaseClassifier:
    def fit(self, train_df, val_df, test_df):
        raise NotImplementedError

    def evaluate(self, test_df):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class HuggingFaceClassifier(BaseClassifier):
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=2
        )

    def preprocess_function(self, examples):
        return self.tokenizer(
            examples["medical_report"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
            return_attention_mask=True,
        )

    def split_datasets(self, df):
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["HeartDisease"]
        )
        train_df, val_df = train_test_split(
            train_df, test_size=0.25, random_state=42, stratify=train_df["HeartDisease"]
        )
        return train_df, val_df, test_df

    def tokenize_datasets(self, train_df, val_df, test_df):
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        tokenized_train_dataset = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_val_dataset = val_dataset.map(self.preprocess_function, batched=True)
        tokenized_test_dataset = test_dataset.map(self.preprocess_function, batched=True)

        tokenized_train_dataset = tokenized_train_dataset.map(
            lambda examples: {"labels": examples["HeartDisease"]}
        )
        tokenized_val_dataset = tokenized_val_dataset.map(
            lambda examples: {"labels": examples["HeartDisease"]}
        )
        tokenized_test_dataset = tokenized_test_dataset.map(
            lambda examples: {"labels": examples["HeartDisease"]}
        )
        return tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset

    def fit(self, train_df, val_df, test_df):
        tokenized_train_dataset, tokenized_val_dataset, _ = self.tokenize_datasets(
            train_df, val_df, test_df
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        training_args = TrainingArguments(
            output_dir="my_awesome_model",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()[:, 1]
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary"
        )
        auc = roc_auc_score(labels, probs)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }

    def evaluate(self, test_df):
        _, _, tokenized_test_dataset = self.tokenize_datasets(None, None, test_df)
        trainer = Trainer(model=self.model, compute_metrics=self.compute_metrics)
        eval_results = trainer.evaluate(tokenized_test_dataset)
        return eval_results

    def predict(self, X):
        tokenized_X = self.tokenizer(X, truncation=True, padding=True, return_tensors="pt")
        test_dataset = Dataset.from_dict({k: v.numpy() for k, v in tokenized_X.items()})
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        eval_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)

        self.model.eval()
        predictions = []
        probabilities = []

        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                preds = logits.argmax(dim=-1).cpu().numpy()
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

                predictions.extend(preds)
                probabilities.extend(probs)

        return predictions, probabilities


class SetFitClassifier(BaseClassifier):
    def init(self, model_checkpoint="sentence-transformers/paraphrase-mpnet-base-v2"):
        self.model_name = model_checkpoint
        self.model = SetFitModel.from_pretrained(self.model_name)

    def fit(self, train_df, val_df, test_df):
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            loss_class=CosineSimilarityLoss,
            metric="accuracy",
            batch_size=16,
            num_iterations=20,
            num_epochs=3,
            column_mapping={"sentence": "text", "label": "label"},
        )
        trainer.train()

    def evaluate(self, test_df):
        test_dataset = Dataset.from_pandas(test_df)
        X_test_list = test_dataset["medical_report"].to_list()
        y_test = test_dataset[test_df.columns[-1]].to_list()
        y_pred, y_prob = self.predict(X_test_list)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary"
        )
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": roc_auc,
        }

    def get_predictions(self, texts):
        probas = self.model.predict_proba(texts, as_numpy=True)
        return np.array([np.argmax(pred) for pred in probas])

    def predict_proba(self, x_test):
        with torch.no_grad():
            probabilities = self.model.predict_proba(x_test, as_numpy=True)
        return probabilities

    def predict(self, X):
        y_pred = self.get_predictions(X)
        y_prob = [probs[1] for probs in self.predict_proba(X)]
        return y_pred, y_prob


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # pltting modulst
    from plots import plot_roc_curve

    DATASET = ""  # NAME OF DATASET that contain medical_report and HeartDisease columns, where HeartDisease is the target column
    df = pd.read_csv(f"/content/drive/My Drive/{DATASET}")[["medical_report", "HeartDisease"]]

    # Main function to run the whole process
    def split_datasets(df):
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["HeartDisease"]
        )
        train_df, val_df = train_test_split(
            train_df, test_size=0.25, random_state=42, stratify=train_df["HeartDisease"]
        )
        return train_df, val_df, test_df

    def evaluate_and_plot(classifier, test_df, model_name, dataset_name):
        eval_results = classifier.evaluate(test_df)
        fpr, tpr, roc_auc = eval_results["fpr"], eval_results["tpr"], eval_results["auc"]
        plot_roc_curve(
            fpr, tpr, roc_auc, test_df["HeartDisease"].to_list(), model_name, dataset_name
        )
        return eval_results

    heart_disease_classifiers = [
        ("xlm-roberta-large", HuggingFaceClassifier),
        ("roberta-base", HuggingFaceClassifier),
        ("bert-base-cased", HuggingFaceClassifier),
        ("dmis-lab/biobert-v1.1", HuggingFaceClassifier),
        ("SetFit", SetFitClassifier),
    ]

    model_scores = []
    model_names = [model_name for model_name, _ in heart_disease_classifiers]
    plt.figure(figsize=(10, 8))

    for model_name, Classifier in heart_disease_classifiers:
        print(f"Processing model: {model_name}")

        classifier = Classifier(model_name)
        train_df, val_df, test_df = split_datasets(df)
        classifier.fit(train_df, val_df, test_df)
        eval_results = evaluate_and_plot(classifier, test_df, model_name, DATASET)

        model_scores.append(
            (
                model_name,
                eval_results["accuracy"],
                eval_results["precision"],
                eval_results["recall"],
                eval_results["f1"],
                eval_results["auc"],
            )
        )

    print(model_scores)
