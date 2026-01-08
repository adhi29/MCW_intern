import torch
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time

print("InceptionV3 Inference Pipeline - ImageNet Validation Set")

print("\n[Step 1] Loading InceptionV3 Model...")
model = models.inception_v3(pretrained=True)
model.eval()
print("✓ InceptionV3 loaded successfully")
print(f"  - Parameters: ~23 million")
print(f"  - Pretrained on: ImageNet-1K")

print("\n[Step 2] Setting up Preprocessing Pipeline...")
preprocess = transforms.Compose([
    transforms.Resize(342),           # Resize shorter side to 342
    transforms.CenterCrop(299),       # Center crop to 299x299
    transforms.ToTensor(),            # Convert to tensor [0, 1]
    transforms.Normalize(             # Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
print("✓ Preprocessing configured")
print("  - Input size: 299x299")
print("  - Normalization: ImageNet statistics")

print("\n[Step 3] Loading ImageNet Validation Dataset...")
val_dataset = datasets.ImageFolder(
    root='/Users/adhi/Desktop/Multicoreware/quantization/ILSVRC2012_img_val',
    transform=preprocess
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

print(f"✓ Dataset loaded successfully")
print(f"  - Dataset: ImageNet-1K Validation Set")
print(f"  - Total images: {len(val_dataset)}")
print(f"  - Number of classes: {len(val_dataset.classes)}")
print(f"  - Batch size: 32")
print(f"  - Total batches: {len(val_loader)}")

print("\n[Step 4] Defining Inference Function...")

def run_inference(model, dataloader, device='cpu', use_fp16=False):
   
    model.to(device)
    model.eval()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    batch_times = []
    
    precision = "FP16" if use_fp16 else "FP32"
    print(f"\n  Device: {device.upper()}")
    print(f"  Precision: {precision}")
    print(f"  Running inference...")
    print(f"  Progress: ", end='')
    
    start_total = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            batch_start = time.time()
            
            # Move to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Convert to FP16 if needed
            if use_fp16:
                images = images.half()
            
            # Forward pass
            outputs = model(images)
            
            # Top-1 accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct_top1 += (predicted == labels).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
            correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            total += labels.size(0)
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Progress indicator every 50 batches
            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - start_total
                eta = (elapsed / (batch_idx + 1)) * (len(dataloader) - batch_idx - 1)
                print(f"\n  [{batch_idx + 1}/{len(dataloader)}] "
                      f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s", end='')
    
    total_time = time.time() - start_total
    
    # Calculate metrics
    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total
    avg_time_per_image = (sum(batch_times) / len(batch_times)) / 32 * 1000  # ms per image
    
    print(f"\n\n  {'=' * 60}")
    print(f"  RESULTS")
    print(f"  {'=' * 60}")
    print(f"  Total images processed: {total}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per image: {avg_time_per_image:.2f}ms")
    print(f"  Throughput: {total/total_time:.2f} images/sec")
    print(f"  {'=' * 60}")
    print(f"  Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"  Top-5 Accuracy: {top5_accuracy:.2f}%")
    print(f"  {'=' * 60}")
    
    return top1_accuracy, top5_accuracy, avg_time_per_image

print("✓ Inference function defined")

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("STARTING INFERENCE")
    print("=" * 70)
    
    # Determine available device
    if torch.cuda.is_available():
        device = 'cuda'
        print("\n✓ CUDA GPU detected")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("\n✓ Apple MPS (Metal) detected")
    else:
        device = 'cpu'
        print("\n⚠️  No GPU detected, using CPU")
    
    # Determine if FP16 should be used
    use_fp16 = False
    
    if device == 'cuda':
        # CUDA has excellent FP16 support
        try:
            model = model.half()
            use_fp16 = True
            print("✓ Model converted to FP16 for CUDA")
        except Exception as e:
            print(f"⚠️  FP16 conversion failed: {e}")
            print("   Falling back to FP32")
            
    elif device == 'mps':
        # MPS has FP16 support but can be inconsistent
        try:
            model = model.half()
            use_fp16 = True
            print("✓ Model converted to FP16 for MPS")
            print("   Note: MPS FP16 support may vary by operation")
        except Exception as e:
            print(f"⚠️  FP16 conversion failed on MPS: {e}")
            print("   Falling back to FP32")
            
    else:
        # CPU - FP16 is not beneficial
        print("ℹ️  Using FP32 on CPU (FP16 not beneficial without GPU)")
    
    # Run inference
    print(f"\n>>> Running inference on {device.upper()} with {'FP16' if use_fp16 else 'FP32'}...")
    
    top1_acc, top5_acc, avg_time = run_inference(model, val_loader, device=device, use_fp16=use_fp16)
    

    print("\n" + "=" * 70)
    print("INFERENCE PIPELINE COMPLETED")
    print("=" * 70)
    print(f"\nModel: InceptionV3 (pretrained)")
    print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
    print(f"Dataset: ImageNet-1K Validation Set")
    print(f"Total images: {len(val_dataset)}")
    print(f"Classes: {len(val_dataset.classes)}")
    print(f"Device: {device.upper()}")
    
    print(f"\n{'=' * 70}")
    print("ACCURACY RESULTS")
    print('=' * 70)
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"Average inference time: {avg_time:.2f}ms per image")
    print("\nEXPECTED PERFORMANCE (from PyTorch documentation)")
    print('=' * 70)
    print("Top-1 Accuracy: ~77.3%")
    print("Top-5 Accuracy: ~93.5%")
    print("\nNote: FP16 may cause slight accuracy differences (~0.1-0.5%)")
    print("      but should provide significant speed improvements on GPU")