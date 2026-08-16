import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import os

# =============================
# CONFIGURATION
# =============================

CLASSES = ["Mobile phone"]
MAX_SAMPLES = 200  # set to None for full dataset
EXPORT_DIR = "./MyMobilePhoneDataset"
SPLITS = ["train", "validation"]

# =============================
# DOWNLOAD + MERGE DATASETS
# =============================

datasets = []
for split in SPLITS:
    print(f"\n📥 Downloading Open Images V7 split: {split}")
    ds = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=CLASSES,
        max_samples=MAX_SAMPLES,
    )

    # Keep only target classes
    ds = ds.filter_labels("ground_truth", F("label").is_in(CLASSES))
    datasets.append(ds)

print("\n🔄 Merging splits into a single dataset...")

# ✅ Works on all versions
merged_dataset = fo.Dataset()
for ds in datasets:
    merged_dataset.add_collection(ds)

merged_dataset.name = "open-images-v7-mobilephone"
print(f"✅ Merged dataset created with {len(merged_dataset)} samples")

# =============================
# EXPORT TO CUSTOM FOLDER (COCO)
# =============================

os.makedirs(EXPORT_DIR, exist_ok=True)

print(f"\n💾 Exporting dataset to: {EXPORT_DIR}")
merged_dataset.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
)

print("\n✅ Export complete!")
print(f"Images + annotations saved to: {os.path.abspath(EXPORT_DIR)}")

# =============================
# OPTIONAL: VISUALIZE IN FIFTYONE APP
# =============================

# Uncomment to explore the dataset interactively
session = fo.launch_app(merged_dataset)
session.wait()
