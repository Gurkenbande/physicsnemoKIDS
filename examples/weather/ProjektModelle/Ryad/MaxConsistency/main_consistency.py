import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from tqdm import tqdm
from consistency_model import ConsistencyDownscalingModel, ConsistencyLoss
from hrrr_mini_dataset import create_dataloaders

print("="*60)
print(" Consistency Model Training ")
print("="*60)

pdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
STATS_PATH = "/home/s458614/data/stats.json"
DATA_PATH = "/home/s458614/data/hrrr_mini_train.nc"
OUTPUT_DIR = r"C:\Users\ryadw\OneDrive\Desktop\outputs"
BATCH_SIZE = 8
NUM_EPCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 0
MODEL_CHANNELS = 128
NUM_STEPS = 18
SAVE_FREQ = 5
PRINT_FREQ = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
print(f"Data: {DATA_PATH}")
print(f"Stats: {STATS_PATH}")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
if not os.path.exists(STATS_PATH):
    raise FileNotFoundError(f"Stats file not found: {STATS_PATH}")

train_loader, val_loader, test_loader = create_dataloaders(
    data_path=DATA_PATH,
    stats_path=STATS_PATH,
    batch_size=BATCH_SIZE,
    train_ratio=0.9,
    val_ratio=0.05,
    num_workers=NUM_WORKERS,
)
print(f"trainieren: {len(train_loader)}")
print(f"val: {len(val_loader)}")
print(f"testen: {len(test_loader)}")

model = ConsistencyDownscalingModel(
    n_input_features=28,
    n_output_vars=4,
    encoding_dim=128,
    model_channels=MODEL_CHANNELS,
    channel_mult=(1, 2, 2, 4),
    num_res_blocks=2,
    attention_resolutions=(16, 8),
    dropout=0.1,
    low_res_size=(8, 8),
    high_res_size=(64, 64),
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    num_steps=NUM_STEPS,
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parametern: {num_params:,}")
loss_fn = ConsistencyLoss(model)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5,
    betas=(0.9, 0.999)
)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
    eta_min=1e-6
)
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    print(f"\n Epoch {epoch+1}/{NUM_EPOCHS}")
    model.train()
    train_loss = 0.0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        optimizer.zero_grad()
        loss = loss_fn(x_flat, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        if batch_idx % PRINT_FREQ == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.6f}")
    
    avg_train_loss = train_loss / len(train_loader)
    print(f"\n Training Loss: {avg_train_loss:.6f}")
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation"):
            x = x.to(device)
            y = y.to(device)
            
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            
            y_pred = model.sample(x_flat, num_steps=1)
            
            loss = F.mse_loss(y_pred, y)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Val Loss: {avg_val_loss:.6f}")
    
    lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"LR: {current_lr:.6e}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoints", "best_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
        }, checkpoint_path)
        print(f"bester Modell (val_loss: {best_val_loss:.6f})")
    
    if (epoch + 1) % SAVE_FREQ == 0:
        checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoints", f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        print(f"checkpoint: epoch_{epoch+1}.pt")

print("\n" + "="*60)
print("Training complleted")
print(f"bester val Loss: {best_val_loss:.6f}")

model.eval()
test_loss = 0.0
predictions = []
targets = []

with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Testing"):
        x = x.to(device)
        y = y.to(device)
        
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        y_pred = model.sample(x_flat, num_steps=1)
        
        loss = F.mse_loss(y_pred, y)
        test_loss += loss.item()
        
        predictions.append(y_pred.cpu())
        targets.append(y.cpu())

avg_test_loss = test_loss / len(test_loader)
print(f"\n Test Loss: {avg_test_loss:.6f}")

predictions = torch.cat(predictions, dim=0)
targets = torch.cat(targets, dim=0)

results_path = os.path.join(OUTPUT_DIR, "test_results.pt")
torch.save({
    'predictions': predictions,
    'targets': targets,
    'test_loss': avg_test_loss,
}, results_path)

print(f"results: {results_path}")