Model in HuggingFace : https://huggingface.co/Vishnu-add/finetuned-indian-food

Dataset in HuggingFace : https://huggingface.co/datasets/Vishnu-add/indian_food_images

Colab notebook link : https://colab.research.google.com/drive/136E39QrIg1hXPmixVu90mY3KOHx0MPAm?authuser=3#scrollTo=NfFH9eLMAdCX

# Image Classification for Food Recognition

This project focuses on the classification of food images using deep learning techniques. It utilizes the Vision Transformer (ViT) model and Hugging Face's Transformers library to identify different types of Indian food from images.

## Table of Contents

1. [Installation](#installation)
2. [Data Collection](#data-collection)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Inference](#inference)
7. [Pipeline API](#pipeline-api)

## Installation

To get started, install the necessary libraries and tools:

```bash
!pip install -q datasets transformers
!sudo apt -qq install git-lfs
!git config --global credential.helper store
!pip install transformers[torch] accelerate>=0.20.1
```

## Data Collection

First, collect and unzip the dataset:

```python
from google.colab import drive
drive.mount('/content/drive')

!unzip "/content/drive/MyDrive/Colab Notebooks/CODING RAJA/Image Classification/archive.zip" -d /tmp/foodimg
from datasets import load_dataset
ds = load_dataset("imagefolder", data_dir="/tmp/foodimg")
ds = ds['train']

data = ds.train_test_split(test_size=0.15)
data.push_to_hub("Vishnu-add/indian_food_images")
```

## Data Preprocessing

Preprocess the data by defining transformations and applying them:

```python
from transformers import ViTFeatureExtractor
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
)

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
train_transforms = Compose([
    RandomResizedCrop((224, 224)),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize,
])

val_transforms = Compose([
    Resize((224, 224)),
    CenterCrop((224, 224)),
    ToTensor(),
    normalize,
])

def preprocess_train(example_batch):
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    example_batch["pixel_values"] = [
        val_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

train_ds = data['train']
val_ds = data['test']
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)
```

## Model Training

Train the Vision Transformer model:

```python
from transformers import ViTForImageClassification, TrainingArguments, Trainer

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(data['train'].features['label'].names),
    id2label={str(i): c for i, c in enumerate(data['train'].features['label'].names)},
    label2id={c: str(i) for i, c in enumerate(data['train'].features['label'].names)}
)

training_args = TrainingArguments(
    'finetuned-indian-food',
    per_device_train_batch_size=32,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=True,
    report_to='tensorboard',
    load_best_model_at_end=True,
    hub_strategy="end"
)

from datasets import load_metric
import numpy as np
import torch

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

trainer = Trainer(
    model,
    training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
```

## Model Evaluation

Evaluate the model:

```python
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
```

## Inference

Run inference on a single image:

```python
from PIL import Image
import requests
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

url = 'https://datasets-server.huggingface.co/assets/Vishnu-add/indian_food_images/--/5525ce321b342f4a644a70283b5166df15d6ae9a/--/default/train/1/image/image.jpg'
image = Image.open(requests.get(url, stream=True).raw)

repo_name = "Vishnu-add/finetuned-indian-food"
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_name)
model = AutoModelForImageClassification.from_pretrained(repo_name)

encoding = feature_extractor(image.convert("RGB"), return_tensors="pt")
with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

## Pipeline API

Use the pipeline API for image classification:

```python
from transformers import pipeline

pipe = pipeline("image-classification", "Vishnu-add/finetuned-indian-food")

image = Image.open(requests.get(url, stream=True).raw)
pipe(image)
```
