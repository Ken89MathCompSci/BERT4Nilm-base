import torch
import numpy as np
from dataset import REDD_LF_Dataset
from dataloader import NILMDataloader
from trainer import Trainer
from model import BERT4NILM
from config import *
from utils import set_template
import argparse
import os

def debug_predictions(args):
    # Set up dataset
    args.validation_size = 1.0
    args.house_indicies = [1]  # Test house
    dataset = REDD_LF_Dataset(args)
    x_mean, x_std = dataset.get_mean_std()
    stats = (x_mean, x_std)

    print("Dataset loaded with {} samples".format(len(dataset.x)))
    print("Appliance names: {}".format(args.appliance_names))
    print("Thresholds: {}".format([args.threshold[app] for app in args.appliance_names]))
    print("Min on: {}".format([args.min_on[app] for app in args.appliance_names]))
    print("Min off: {}".format([args.min_off[app] for app in args.appliance_names]))

    # Check ground truth status
    x, y, status = dataset.get_data()
    print("\nGround truth analysis:")
    for i, app in enumerate(args.appliance_names):
        on_samples = np.sum(status[:, i])
        total_samples = len(status)
        print("{}: {}/{} samples ON ({:.2f}%)".format(app, int(on_samples), total_samples, 100*on_samples/total_samples))

        # Check power levels when appliance is supposedly ON
        on_power = y[status[:, i] == 1, i]
        if len(on_power) > 0:
            print("  Power when ON: mean={:.1f}W, max={:.1f}W, min={:.1f}W".format(np.mean(on_power), np.max(on_power), np.min(on_power)))

    # Load model
    folder_name = '-'.join(args.appliance_names)
    export_root = 'experiments/' + args.dataset_code + '/' + folder_name
    model_path = os.path.join(export_root, 'best_acc_model.pth')

    if not os.path.exists(model_path):
        print("Model not found at {}".format(model_path))
        return

    model = BERT4NILM(args)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.to(args.device)
    model.eval()

    # Create dataloader
    dataloader = NILMDataloader(args, dataset, bert=False)
    _, test_loader = dataloader.get_dataloaders()

    print("\nModel predictions analysis:")
    all_pred_status = []
    all_true_status = []
    all_pred_power = []
    all_true_power = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 5:  # Just check first few batches
                break

            seqs, labels_energy, status = batch
            seqs, labels_energy, status = seqs.to(args.device), labels_energy.to(args.device), status.to(args.device)

            logits = model(seqs)
            pred_energy = torch.clamp(logits * torch.tensor([args.cutoff[app] for app in args.appliance_names]).to(args.device), 0, torch.tensor([args.cutoff[app] for app in args.appliance_names]).to(args.device))
            pred_status = (pred_energy >= torch.tensor([args.threshold[app] for app in args.appliance_names]).to(args.device)).float()

            all_pred_status.append(pred_status.cpu().numpy())
            all_true_status.append(status.cpu().numpy())
            all_pred_power.append(pred_energy.cpu().numpy())
            all_true_power.append(labels_energy.cpu().numpy())

    # Analyze predictions
    pred_status = np.concatenate(all_pred_status, axis=0)
    true_status = np.concatenate(all_true_status, axis=0)
    pred_power = np.concatenate(all_pred_power, axis=0)
    true_power = np.concatenate(all_true_power, axis=0)

    for i, app in enumerate(args.appliance_names):
        pred_on = np.sum(pred_status[:, i])
        true_on = np.sum(true_status[:, i])
        total = len(pred_status)

        print("\n{} predictions:".format(app))
        print("  Predicted ON: {}/{} samples ({:.2f}%)".format(int(pred_on), total, 100*pred_on/total))
        print("  True ON: {}/{} samples ({:.2f}%)".format(int(true_on), total, 100*true_on/total))

        # Check power predictions when model predicts ON
        pred_on_power = pred_power[pred_status[:, i] == 1, i]
        if len(pred_on_power) > 0:
            print("  Predicted power when ON: mean={:.1f}W, max={:.1f}W".format(np.mean(pred_on_power), np.max(pred_on_power)))

        # Check if model ever predicts reasonable power levels
        reasonable_power = pred_power[pred_power[:, i] > args.threshold[app], i]
        if len(reasonable_power) > 0:
            print("  Power predictions above threshold: {} samples, mean={:.1f}W".format(len(reasonable_power), np.mean(reasonable_power)))

if __name__ == "__main__":
    # Create args similar to training
    args = argparse.Namespace(
        dataset_code='redd_lf',
        appliance_names=['refrigerator'],  # Change this for different appliances
        device='cpu',
        validation_size=1.0,
        batch_size=128,
        window_size=480,
        window_stride=120,
        normalize='mean',
        cutoff=None,
        threshold=None,
        min_on=None,
        min_off=None,
        output_size=1,
        mask_prob=0.25,
        sampling='6s',  # Required by dataset
        denom=2000      # Required by trainer
    )

    # Set template parameters
    set_template(args)
    debug_predictions(args)
