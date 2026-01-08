import os
import shutil
import urllib.request

print("=" * 70)
print("Organizing ImageNet Validation Set")
print("=" * 70)

# Download the validation ground truth labels
print("\n[Step 1] Downloading validation labels...")
labels_url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
urllib.request.urlretrieve(labels_url, "imagenet_class_index.json")

# Also get the validation ground truth file
val_labels_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt"
urllib.request.urlretrieve(val_labels_url, "val_labels.txt")
print("✓ Labels downloaded")

# Read validation labels
print("\n[Step 2] Reading validation labels...")
with open('val_labels.txt', 'r') as f:
    labels = [line.strip() for line in f]
print(f"✓ Found {len(labels)} labels")

# Get all validation images
val_dir = '/Users/adhi/Desktop/Multicoreware/quantization/ILSVRC2012_img_val'
print(f"\n[Step 3] Scanning validation directory: {val_dir}")

if not os.path.exists(val_dir):
    print(f"❌ Error: Directory {val_dir} not found!")
    print("Please extract ILSVRC2012_img_val.tar to data/imagenet/val first")
    exit()

images = sorted([f for f in os.listdir(val_dir) if f.endswith('.JPEG')])
print(f"✓ Found {len(images)} images")

# Organize images into class folders
print("\n[Step 4] Organizing images into class folders...")
for idx, (img, label) in enumerate(zip(images, labels)):
    # Create class folder
    class_dir = os.path.join(val_dir, label)
    os.makedirs(class_dir, exist_ok=True)
    
    # Move image
    src = os.path.join(val_dir, img)
    dst = os.path.join(class_dir, img)
    
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
    
    # Progress indicator
    if (idx + 1) % 1000 == 0:
        print(f"  Processed {idx + 1}/{len(images)} images...")

print(f"\n✓ All {len(images)} images organized into {len(set(labels))} class folders")

# Verify structure
print("\n[Step 5] Verifying folder structure...")
class_folders = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
print(f"✓ Created {len(class_folders)} class folders")

# Count images per folder (sample check)
sample_class = class_folders[0]
sample_path = os.path.join(val_dir, sample_class)
sample_count = len([f for f in os.listdir(sample_path) if f.endswith('.JPEG')])
print(f"✓ Sample check - Folder '{sample_class}' contains {sample_count} images")

print("\n" + "=" * 70)
print("✓ ImageNet Validation Set Organization Complete!")
print("=" * 70)
print(f"\nFinal structure:")
print(f"  {val_dir}/")
print(f"  ├── n01440764/ (class folder)")
print(f"  ├── n01443537/ (class folder)")
print(f"  └── ... ({len(class_folders)} class folders total)")
print(f"\nYou can now run the inference pipeline!")
