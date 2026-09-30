import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
import numpy as np
import traceback

DATASET_PATH = "/home/d/durmusy/Desktop/GIT/new/uedrlhf/data_v3/xland_minigrid/ruleset_difficult/difficulty_feedback_low/compiled_gemini_3_5_flash_len_21973"
ds = load_from_disk(DATASET_PATH)

columns = ['env_0_image', 'env_1_image', 'preference']
ds.set_format(type="numpy", columns=columns)

dataloader = DataLoader(
    ds,
    batch_size=128,
    shuffle=False,
    drop_last=False,
    num_workers=0
)

print("Starting iteration...")
try:
    for i, batch in enumerate(dataloader):
        print(f"Batch {i} loaded successfully.")
        break
except Exception as e:
    print("Error encountered:")
    traceback.print_exc()
