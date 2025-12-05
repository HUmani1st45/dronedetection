import os
from glob import glob
from typing import List, Dict, Tuple
import itertools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils as utils # For gradient clipping

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms.functional as TF
from PIL import Image

# ==============================================================================
# 1. GRADIENT REVERSAL LAYER (GRL)
# ==============================================================================

class GradientReverseFn(autograd.Function):
    """Handles the forward (identity) and backward (gradient reversal) pass."""
    @staticmethod
    def forward(ctx, x, lambda_):
        # Save the dynamic lambda for the backward pass
        ctx.lambda_ = lambda_ 
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse and scale gradient
        return -ctx.lambda_ * grad_output, None

class GradientReverse(nn.Module):
    """Module wrapper for GRL, accepts lambda in forward for dynamic scheduling."""
    def __init__(self):
        super().__init__()

    def forward(self, x, lambda_):
        # Pass the dynamically calculated lambda_ directly to the function
        return GradientReverseFn.apply(x, lambda_)

# ==============================================================================
# 2. DA-FASTER R-CNN MODEL
# ==============================================================================

class DAFasterRCNN_Global_Stable(nn.Module):
    """
    Domain-Adaptive Faster R-CNN with image-level domain classification 
    using FPN features and a GRL.
    """
    def __init__(self, num_classes, lambda_da_max=1.0, da_warmup_steps=1000):
        super().__init__()
        self.lambda_da_max = lambda_da_max
        self.da_warmup_steps = da_warmup_steps 
        self.current_step = 0 

        # ----------- BASE DETECTOR (ResNet50-FPN) -------------
        # Custom anchors for drone detection (small objects)
        anchor_generator = AnchorGenerator(
            sizes=((8, 16, 32), (16, 32, 64), (32, 64, 128), (64, 128, 256), (128, 256, 512)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        self.detector = fasterrcnn_resnet50_fpn(
            weights=None,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator
        )

        self.grl = GradientReverse()
        self.im_domain_head = None # Lazily initialized

    def set_current_step(self, step):
        """Allows external update of the current iteration/step count."""
        self.current_step = step

    def warmup_lambda(self):
        """Gradually increase lambda_da over first steps (linear schedule)."""
        if self.current_step >= self.da_warmup_steps:
            return self.lambda_da_max
        
        progress = self.current_step / self.da_warmup_steps
        return progress * self.lambda_da_max

    def _build_domain_head(self, feature_dict):
        """Initializes the domain classifier based on the feature dimensions."""
        num_scales = len(feature_dict)
        feat_dim = next(iter(feature_dict.values())).shape[1] # Expected 256 for FPN
        input_dim = num_scales * feat_dim

        # Domain Classifier Head: simple MLP for binary classification (Source vs. Target)
        self.im_domain_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        ).to(next(self.parameters()).device)

    def forward(self, images, targets=None, domain=None):
        if self.training and domain is None:
            raise ValueError("Need domain='source' or 'target' during training.")

        # In inference, use the standard detector forward
        if not self.training:
            self.detector.eval()
            with torch.no_grad():
                return self.detector(images)
        
        # --- TRAINING MODE ---

        # Forward pass through backbone to get FPN features
        images_t, targets_t = self.detector.transform(images, targets) 
        features = self.detector.backbone(images_t.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # Lazy initialization check
        if self.im_domain_head is None:
            self._build_domain_head(features)

        det_losses = {}
        # DETECTION TASK LOSS (Only calculated on labeled SOURCE data)
        if domain == "source":
            # Call the detector's forward method when targets are provided to get losses
            det_outputs = self.detector.forward(images, targets)
            det_losses.update(det_outputs)

        # --- DOMAIN ADAPTATION LOSS ---
        da_losses = {}
        
        # Domain label (0=source, 1=target)
        dom_label = 0 if domain == "source" else 1
        dom_label_tensor = torch.full(
            (len(images),), dom_label, dtype=torch.long, device=images[0].device
        )

        # Global pooled features (concatenate features from all FPN levels)
        pooled = [
            F.adaptive_avg_pool2d(f, 1).flatten(1)
            for f in features.values()
        ]
        im_feat = torch.cat(pooled, dim=1)

        # GRL with warm-up
        lambda_da = self.warmup_lambda()
        # Apply GRL: gradient is reversed and scaled by lambda_da
        rev_feat = self.grl(im_feat, lambda_da)

        logits = self.im_domain_head(rev_feat)
        da_losses["loss_da_im"] = F.cross_entropy(logits, dom_label_tensor)

        losses = {}
        losses.update(det_losses)
        losses.update(da_losses)
        return losses

# ==============================================================================
# 3. DATASET AND DATALOADER UTILITIES
# ==============================================================================

class YoloTxtDetectionDataset(Dataset):
    """Loads images and converts YOLO format labels (normalized cx,cy,w,h) 
    to TorchVision format (absolute x1,y1,x2,y2)."""
    def __init__(self, img_dir: str, label_dir: str, transforms=None):
        self.img_paths = sorted(
            [p for p in glob(os.path.join(img_dir, "*")) if p.lower().endswith((".jpg", ".png", ".jpeg"))]
        )
        self.label_dir = label_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def _load_yolo_labels(self, img_path: str, w: int, h: int):
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, base + ".txt")

        boxes = []
        labels = []

        if not os.path.exists(label_path):
            # Crucial: return empty tensors if no labels found
            return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.int64)

        with open(label_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line: continue
                parts = line.split()
                # YOLO format: class_id cx cy w h (normalized to 0-1)
                cls = int(parts[0]) 
                x_c, y_c, bw, bh = map(float, parts[1:5])

                # Convert normalized cx, cy, w, h → absolute pixel xyxy
                x_c, y_c, bw, bh = x_c * w, y_c * h, bw * w, bh * h
                x1 = x_c - bw / 2
                y1 = y_c - bh / 2
                x2 = x_c + bw / 2
                y2 = y_c + bh / 2

                boxes.append([x1, y1, x2, y2])
                # IMPORTANT: Faster R-CNN reserves class ID 0 for background. 
                # If YOLO is 0-indexed, shift classes by 1.
                labels.append(cls + 1) 

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        return boxes, labels

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes, labels = self._load_yolo_labels(img_path, w, h)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            # Required for TorchVision: area and iscrowd
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if boxes.numel() > 0 else torch.tensor([], dtype=torch.float32),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        # Convert PIL Image to Tensor
        img = TF.to_tensor(img)
        return img, target

def det_collate(batch):
    """Custom collate function required by TorchVision detection models."""
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)

# ==============================================================================
# 4. TRAINING FUNCTION
# ==============================================================================

def train_da_stable(
    model,
    source_loader,
    target_loader,
    optimizer,
    num_epochs=10,
    da_weight=0.001,
    device="cuda"
):
    model.train()
    # Cycle the target loader so it runs as long as the source loader
    target_iter = itertools.cycle(target_loader) 
    global_step = 0 
    
    # Initialize total_loss outside the loop for safe use in checkpointing on epoch 0
    total_loss = torch.tensor(0.0, device=device) 

    print(f"Starting training on device: {device}")
    print(f"Total batches per epoch: {len(source_loader)}")

    for epoch in range(num_epochs):
        
        # -----------------------------------------------
        # 1. BATCH LOOP (INNER LOOP)
        # -----------------------------------------------
        for i, (src_imgs, src_tgts) in enumerate(source_loader):
            
            # Update step and lambda schedule
            global_step += 1
            model.set_current_step(global_step)

            src_imgs = [img.to(device) for img in src_imgs]
            src_tgts = [{k: v.to(device) for k, v in t.items()} for t in src_tgts]

            # --- SOURCE FORWARD PASS (Task Loss + DA Loss) ---
            src_loss_dict = model(src_imgs, src_tgts, domain="source")

            # Aggregate detection task losses
            det_loss = (
                src_loss_dict.get("loss_classifier", torch.tensor(0.0, device=device)) +
                src_loss_dict.get("loss_box_reg", torch.tensor(0.0, device=device)) +
                src_loss_dict.get("loss_objectness", torch.tensor(0.0, device=device)) +
                src_loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0, device=device))
            )
            da_src = src_loss_dict.get("loss_da_im", torch.tensor(0.0, device=device))

            # --- TARGET FORWARD PASS (DA Loss ONLY) ---
            tgt_imgs, _ = next(target_iter)
            tgt_imgs = [img.to(device) for img in tgt_imgs]

            tgt_loss_dict = model(tgt_imgs, targets=None, domain="target")
            da_tgt = tgt_loss_dict.get("loss_da_im", torch.tensor(0.0, device=device))

            # --- TOTAL LOSS & OPTIMIZATION ---
            # Total Loss = Task Loss + DA Weight * (DA Source Loss + DA Target Loss)
            total_loss = det_loss + da_weight * (da_src + da_tgt)

            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping for adversarial stability (CRITICAL)
            utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            
            # --- LOGGING ---
            if global_step % 10 == 0:
                current_lambda = model.warmup_lambda()
                print(
                    f"[Epoch {epoch} Step {global_step}] "
                    f"det={det_loss.item():.4f} "
                    f"DA_src={da_src.item():.4f} "
                    f"DA_tgt={da_tgt.item():.4f} "
                    f"lambda={current_lambda:.6f} "
                    f"total={total_loss.item():.4f}"
                )
        
        # -----------------------------------------------
        # 2. CHECKPOINT SAVING (Runs once per epoch)
        # -----------------------------------------------
        save_path = f"da_frcnn_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss.item(), # Last calculated loss of this epoch
        }, save_path)
        print(f"\nModel checkpoint saved to {save_path}\n")

    print("Training complete.")


# ==============================================================================
# 5. INITIALIZATION AND EXECUTION
# ==============================================================================

if __name__ == '__main__':
    BASE_DATA_PATH = '/content/data_extracted' 

# Source (Virtual/Synthetic)
    virt_img_dir = f"{BASE_DATA_PATH}/virtual/train/images" 
    virt_lbl_dir = f"{BASE_DATA_PATH}/virtual/train/labels"

# Target (Real/Unlabeled)
    real_img_dir = f"{BASE_DATA_PATH}/real_LRDD/train/images" 
    real_lbl_dir = f"{BASE_DATA_PATH}/real_LRDD/train/labels" 

# ... il resto del tuo script da qui in poi usa queste nuove variabili ...
    
    # Classification: Background (0) + 1 Drone Class (1) = 2 classes
    num_classes = 2 
    
    # DA Hyperparameters (Crucial for stability)
    lambda_da_max = 1.0       # Max GRL scaling factor (internally scheduled)
    da_warmup_steps = 1000    # Steps to reach lambda_da_max
    da_weight = 0.001         # External scalar weight for the overall DA Loss
    
    num_epochs = 10
    batch_size = 2 # Use small batches for detection models (adjust based on GPU memory)
    
    # --- DEVICE SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # --- DATA LOADERS ---
    source_dataset = YoloTxtDetectionDataset(virt_img_dir, virt_lbl_dir)
    target_dataset = YoloTxtDetectionDataset(real_img_dir, real_lbl_dir)

    source_loader = DataLoader(
        source_dataset, batch_size=batch_size, shuffle=True, collate_fn=det_collate, num_workers=4
    )
    target_loader = DataLoader(
        target_dataset, batch_size=batch_size, shuffle=True, collate_fn=det_collate, num_workers=4
    )
    
    # --- MODEL & OPTIMIZER ---
    model = DAFasterRCNN_Global_Stable(
        num_classes=num_classes,
        lambda_da_max=lambda_da_max,
        da_warmup_steps=da_warmup_steps
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001, # Low LR is critical for adversarial stability
        momentum=0.9,
        weight_decay=0.0005
    )

    # --- START TRAINING ---
    # To run on the VM using tmux, execute: python3 train_dafrcnn.py
    train_da_stable(
        model, 
        source_loader, 
        target_loader, 
        optimizer, 
        num_epochs=num_epochs, 
        da_weight=da_weight, 
        device=device
    )