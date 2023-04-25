from utils_ import get_instance,ROP_Dataset
import os
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader
import numpy as np
from config import get_config,update_config
import models
from config import _C as config
if __name__=='__main__':
    # Parse arguments
    args = get_config()
    update_config(config,args)
    
    # Create the model and criterion
    model = get_instance(models, config.MODEL.MODEL_NAME,config,
                         num_classes=ROP_Dataset(args.path_tar,split='train').num_classes())
    criterion=torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(
        torch.load(args.save_name))
    model.eval()

    # Create the dataset and data loader
    test_dataset = ROP_Dataset(args.path_tar,split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Test the model and save visualizations
    model.eval()
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for inputs, targets, meta in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    acc = accuracy_score(all_targets, np.round(all_outputs))
    auc = roc_auc_score(all_targets, all_outputs)
    print(f"Finished testing! Test acc {acc:.4f} Test AUC: {auc:.4f} ")

