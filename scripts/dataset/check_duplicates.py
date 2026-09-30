import hashlib
from tqdm import tqdm
from datasets import load_from_disk
from uedrlhf.config import BASE_DIR
ds = load_from_disk(str(BASE_DIR / "dataset_1M/dataset"))

def find_duplicates(dataset):
    hashes = {}
    duplicates = []

    print("Hashing images to find duplicates...")
    for i in tqdm(range(len(dataset))):
        # 1. Get the raw image data
        img = dataset[i]["image"]
        
        # 2. Convert PIL image to raw bytes
        img_bytes = img.tobytes()
        
        # 3. Create a unique hash for these bytes
        img_hash = hashlib.md5(img_bytes).hexdigest()
        
        if img_hash in hashes:
            duplicates.append((i, hashes[img_hash]))
        else:
            hashes[img_hash] = i
            ##print(i)

    return duplicates

dup_list = find_duplicates(ds)

if not dup_list:
    print("No duplicate images found. Your 1M images are unique!")
else:
    print(f"Found {len(dup_list)} duplicate images.")
    print(f"Example: Index {dup_list[0][0]} is a duplicate of Index {dup_list[0][1]}")