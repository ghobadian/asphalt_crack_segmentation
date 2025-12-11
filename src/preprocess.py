import json
import random
import shutil
from collections import defaultdict

def reorganize_and_clean_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, random_seed=37):
    """
    Cleans, unifies, and re-splits the COCO dataset.

    Args:
        source_dir (str): Path to the raw dataset with train/valid/test folders.
        target_dir (str): Path to save the clean and reorganized dataset.
    """
    random.seed(random_seed)

    CATEGORY_MAPPING = {
        'crack': 'crack',
        'Crack': 'crack',
        'Crank': 'crack',
        'objects': 'crack',
        '0': 'crack'
    }
    UNIFIED_CATEGORIES = [{"id": 1, "name": "crack", "supercategory": "defect"}]

    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)

    print("Collecting all images, including positive (crack) and negative (no crack) samples...")
    all_data = []
    for split in ['train', 'valid', 'test']:
        ann_file = os.path.join(source_dir, split, '_annotations.coco.json')
        if not os.path.exists(ann_file):
            continue
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)

        cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        img_id_to_anns = defaultdict(list)
        for ann in coco_data['annotations']:
            img_id_to_anns[ann['image_id']].append(ann)

        for img in coco_data['images']:
            img_anns = img_id_to_anns[img['id']]

            positive_anns = [
                ann for ann in img_anns
                if CATEGORY_MAPPING.get(cat_id_to_name.get(ann['category_id']))
            ]

            all_data.append({
                'image': img,
                'annotations': positive_anns,
                'original_path': os.path.join(source_dir, split, img['file_name'])
            })

    print(f"Found {len(all_data)} total images to include in the new dataset.")

    random.shuffle(all_data)
    total_size = len(all_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_data = all_data[:train_size]
    val_data = all_data[train_size : train_size + val_size]
    test_data = all_data[train_size + val_size:]

    print(f"\nNew Split:")
    print(f"  Training:   {len(train_data)} images")
    print(f"  Validation: {len(val_data)} images")
    print(f"  Test:       {len(test_data)} images")

    def save_split(data_list, split_name):
        new_images, new_annotations = [], []
        ann_counter = 1
        for img_idx, item in enumerate(data_list):
            shutil.copy2(item['original_path'], os.path.join(target_dir, split_name, item['image']['file_name']))

            item['image']['id'] = img_idx
            new_images.append(item['image'])

            for ann in item['annotations']:
                ann['id'] = ann_counter
                ann['image_id'] = img_idx
                ann['category_id'] = 1
                new_annotations.append(ann)
                ann_counter += 1

        new_coco_data = {
            "images": new_images,
            "annotations": new_annotations,
            "categories": UNIFIED_CATEGORIES
        }

        with open(os.path.join(target_dir, split_name, '_annotations.coco.json'), 'w') as f:
            json.dump(new_coco_data, f, indent=2)
        print(f"  Saved {split_name} split with {len(new_images)} images and {len(new_annotations)} annotations.")

    print("\nSaving new splits to target directory...")
    save_split(train_data, 'train')
    save_split(val_data, 'valid')
    save_split(test_data, 'test')

    print(f"\n✅ Dataset reorganization complete. Clean data is in: {target_dir}")


reorganize_and_clean_dataset(source_dir=DATA_RAW_DIR, target_dir=DATA_CLEAN_DIR)


import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

!pip install pycocotools -q

class CrackSegmentationDataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=512):
        self.split_dir = os.path.join(data_dir, split)
        self.img_size = img_size
        ann_file = os.path.join(self.split_dir, '_annotations.coco.json')
        self.coco = COCO(ann_file)
        self.img_ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # --- 1. Load Image ---
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.split_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- 2. Create Mask from Polygons ---
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            # pycocotools function to rasterize a polygon
            mask = np.maximum(self.coco.annToMask(ann), mask)

        # --- 3. Resize ---
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # --- 4. Normalize and Convert to Tensor ---
        image = image.astype(np.float32) / 255.0

        # Ensure mask is binary
        mask = (mask > 0).astype(np.float32)

        # Convert to PyTorch tensors with correct dimensions (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0) # Add channel dimension

        return image_tensor, mask_tensor

# --- Create Datasets and DataLoaders ---
print("Creating Datasets and DataLoaders...")
train_dataset = CrackSegmentationDataset(DATA_CLEAN_DIR, split='train')
valid_dataset = CrackSegmentationDataset(DATA_CLEAN_DIR, split='valid')
test_dataset = CrackSegmentationDataset(DATA_CLEAN_DIR, split='test')

BATCH_SIZE = 14 # Adjust based on GPU memory
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTrain loader: {len(train_loader)} batches of size {BATCH_SIZE}")
print(f"Validation loader: {len(valid_loader)} batches of size {BATCH_SIZE}")
print(f"Test loader: {len(test_loader)} batches of size {BATCH_SIZE}")
print("\n✅ Data loading pipeline is ready.")