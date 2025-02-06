import os
import sys
import json
from typing import Type, TypeVar
from dataclasses import dataclass, field, fields

import torch
import torch.optim as optim
from torch.utils.data import random_split
import torchvision.transforms.v2 as v2

from transformers.optimization import get_cosine_schedule_with_warmup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dpr import dpr_create
from scripts.trainer import Trainer
from utils.data_preprocessing import DPRDataset

T = TypeVar("T")

@dataclass
class Arguments:
    lr: str = field(
        default=1e-4,
        metadata={"help": "The learning rate for the optimizer."},
    )
    
    batch_size: int = field(
        default=32,
        metadata={"help": "The batch size for training."},
    )

    num_epochs: int = field(
        default=100,
        metadata={"help": "The number of epochs to train for."},
    )

    num_patience: int = field(
        default=5,
        metadata={"help": "The number of epochs to wait before early stopping."},
    )

    use_scheduler: bool = field(
        default=True,
        metadata={"help": "Usage scheduler"}
    )

    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "The ratio of warmup steps to total steps."},
    )

    seed: int = field(
        default=42,
        metadata={"help": "The seed to use for reproducibility."},
    )

    checkpoint_dir: str = field(
        default="./checkpoints",
        metadata={"help": "The directory to save model checkpoints."},
    )

    root_dir: str = field(
        default="data",
        metadata={"help": "The root directory for the dataset."},
    )

    train_path: str = field(
        default="data/train_dataset.json",
        metadata={"help": "The path to the training data."},
    )

    val_path: str = field(
        default="data/val_dataset.json",
        metadata={"help": "The path to the validation data."},
    )


def parse_config(config_path: str, cls: Type[T]) -> T:
    """
    Parse a JSON config file and convert it to an instance of the given dataclass.
    
    Args:
        config_path (str): The path to the JSON config file.
        cls (Type[T]): The dataclass type to map the JSON data to.
    
    Returns:
        T: An instance of the dataclass populated with the JSON data.
    """
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    cls_fields = {field.name for field in fields(cls)}
    filtered_data = {k: v for k, v in config_data.items() if k in cls_fields}
    
    return cls(**filtered_data)


def main():
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parse_config(sys.argv[1], Arguments)

    transform_train = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    transform_test = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    train_data = DPRDataset(root_dir=args.root_dir, file_name=args.train_path, transform=transform_train)
    val_test_data = DPRDataset(root_dir=args.root_dir, file_name=args.val_path, transform=transform_test)

    val_size = int(len(val_test_data) * 0.5)
    test_size = len(val_test_data) - val_size

    val_data, test_data = random_split(val_test_data, [val_size, test_size])

    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Test data size: {len(test_data)}")

    q_model, p_model = dpr_create()

    optimzier_grouped_parameters = [
        {"params": q_model.parameters()},
        {"params": p_model.parameters()}
    ]

    optimizer = optim.AdamW(optimzier_grouped_parameters, lr=args.lr, weight_decay=0.05)
    
    if args.use_scheduler:
        total_steps = len(train_data) * args.num_epochs
        warmup_steps = args.warmup_ratio * total_steps
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    else:
        scheduler = None

    trainer = Trainer(
        q_model=q_model,
        p_model=p_model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        scheduler=scheduler,
        **vars(args)
    )

    # trainer.train()
    trainer.test()


if __name__ == "__main__":
    main()
    