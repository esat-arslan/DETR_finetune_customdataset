# DETR Fine-Tuning Project

This project fine-tunes the DETR (DEtection TRansformer) model on a custom dataset using Hugging Face Transformers and PyTorch Lightning. The dataset is in COCO format and includes preprocessing steps for compatibility with the DETR model.

## Project Overview

- **Model**: [DETR (DEtection TRansformer)](https://github.com/facebookresearch/detr)
- **Frameworks**: PyTorch, Hugging Face Transformers, PyTorch Lightning
- **Dataset Format**: COCO JSON annotations
- **Metrics**: mean Average Precision (mAP), Recall

## Installation

Install the required dependencies:

```bash
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q pytorch-lightning
```

## Dataset

The dataset must follow the COCO format, with `_annotations.coco.json` in the dataset directory.

- **Train Data**: `/content/ChessPieces-3/train`
- **Validation Data**: `/content/ChessPieces-3/valid`

A custom `CocoDetection` class handles dataset loading and preprocessing.

### Example

```python
from transformers import DetrImageProcessor

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

train_dataset = CocoDetection(img_folder='/path/to/train', processor=processor)
val_dataset = CocoDetection(img_folder='/path/to/valid', processor=processor, train=False)
```

## Model Training

The DETR model is loaded and fine-tuned using the `Trainer` API from Hugging Face Transformers.

### Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=10,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    remove_unused_columns=False,
    metric_for_best_model="eval_map",
    greater_is_better=True
)
```

### Training Execution

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn
)

trainer.train()
```

## Evaluation

Evaluation is performed using COCO-style metrics, including mAP and Recall. A custom evaluator processes model predictions and computes metrics.

```python
from transformers import Trainer

eval_results = trainer.evaluate()
print(eval_results)
```

## Customization

- Modify `CocoDetection` for custom preprocessing or augmentation.
- Adjust `TrainingArguments` for hyperparameter tuning.
- Use a different backbone or model by replacing `facebook/detr-resnet-50`.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Facebook Research DETR](https://github.com/facebookresearch/detr)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

