import torch
from torch.utils.data import DataLoader
from config import get_config
from utils_ import get_instance, train_epoch, val_epoch,get_optimizer
import os
import models
from utils_ import ROP_Dataset
# Parse arguments
args,config = get_config()

# Initialize the folder
os.makedirs(config.SAVE_DIR,exist_ok=True)
# Init the result file to store the pytorch model and other mid-result
model_path = os.path.join(config.MODEL.SAVE_DIR,config.MODEL.SAVE_NAME)
print(f"model will be stored in {model_path}")

os.makedirs(config.RESULT_PATH,exist_ok=True)
print(f"Intermediate results and pytorch official pretrained models will be stored in {config.RESULT_PATH}")
# Create the model and criterion
model, criterion = get_instance(models, config.MODEL.MODEL_NAME,config.MODEL)
if os.path.isfile(config.BEGIN_CHECKPOINT):
    print(f"loadding the exit checkpoints {config.BEGIN_CHECKPOINT}")
    model.load_state_dict(
    torch.load(args.from_checkpoint))
    
# Creatr optimizer
optimizer = get_optimizer(config, model)


last_epoch = args.configs.TRAIN.BEGIN_EPOCH
if isinstance(args.configs.TRAIN.LR_STEP, list):
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.configs.TRAIN.LR_STEP,
        args.configs.TRAIN.LR_FACTOR, last_epoch-1
    )
else:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.configs.TRAIN.LR_STEP,
        args.configs.TRAIN.LR_FACTOR, last_epoch-1
    )

# Load the datasets #TODO
train_dataset,val_dataset=ROP_Dataset(args.path_tar)
# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
                          shuffle=True, num_workers=config.TRAIN.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
                        shuffle=False, num_workers=config.TRAIN.NUM_WORKERS)

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")

# Set up the optimizer, loss function, and early stopping
early_stop_counter = 0
best_val_loss = float('inf')
total_epoches=args.configs.TRAIN.END_EPOCH

# Training and validation loop
for epoch in range(last_epoch,total_epoches):
    train_loss = train_epoch(model, optimizer, train_loader, criterion, device)
    val_loss, val_acc, val_auc = val_epoch(model, val_loader, criterion, device)

    print(f'Epoch {epoch + 1}/{total_epoches}, '
          f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
          f'Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(),model_path)
        print(f"Model saved in epoch {epoch}")
    else:
        early_stop_counter += 1
        if early_stop_counter >= args.configs.TRAIN.EARLY_STOP:
            print("Early stopping triggered")
            break
