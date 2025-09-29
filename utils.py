import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def get_user_input(args):
    if torch.cuda.is_available():
        args.device = 'cuda:' + input('Input GPU ID: ')
    else:
        args.device = 'cpu'

    dataset_code = {'r': 'redd_lf', 'u': 'uk_dale'}
    args.dataset_code = dataset_code[input(
        'Input r for REDD, u for UK_DALE: ')]

    if args.dataset_code == 'redd_lf':
        app_dict = {
            'r': ['refrigerator'],
            'w': ['washer_dryer'],
            'm': ['microwave'],
            'd': ['dishwasher'],
        }
        args.appliance_names = app_dict[input(
            'Input r, w, m or d for target appliance: ')]

    elif args.dataset_code == 'uk_dale':
        app_dict = {
            'k': ['kettle'],
            'f': ['fridge'],
            'w': ['washing_machine'],
            'm': ['microwave'],
            'd': ['dishwasher'],
        }
        args.appliance_names = app_dict[input(
            'Input k, f, w, m or d for target appliance: ')]

    args.num_epochs = int(input('Input training epochs: '))


def set_template(args):
    args.output_size = len(args.appliance_names)
    if args.dataset_code == 'redd_lf':
        args.window_stride = 120
        args.house_indicies = [1, 2, 3, 4, 5, 6]

        args.cutoff = {
            'aggregate': 6000,
            'refrigerator': 400,
            'washer_dryer': 3500,
            'microwave': 1800,
            'dishwasher': 1200
        }

        args.threshold = {
            'refrigerator': 50,  # Increased from 10W to reduce false positives
            'washer_dryer': 100,
            'microwave': 50,  # Reduced from 200W to capture more microwave usage events
            'dishwasher': 10
        }

        args.min_on = {
            'refrigerator': 5,
            'washer_dryer': 150,
            'microwave': 2,
            'dishwasher': 300
        }

        args.min_off = {
            'refrigerator': 2,
            'washer_dryer': 26,
            'microwave': 5,
            'dishwasher': 300
        }

        args.c0 = {
            'refrigerator': 1e-3,  # Increased from 1e-6 for better refrigerator training
            'washer_dryer': 0.001,
            'microwave': 1.,
            'dishwasher': 1.
        }

    elif args.dataset_code == 'uk_dale':
        args.window_stride = 240
        args.house_indicies = [1, 2, 3, 4, 5]
        
        args.cutoff = {
            'aggregate': 6000,
            'kettle': 3100,
            'fridge': 300,
            'washing_machine': 2500,
            'microwave': 3000,
            'dishwasher': 2500
        }

        args.threshold = {
            'kettle': 2000,
            'fridge': 50,
            'washing_machine': 20,
            'microwave': 200,
            'dishwasher': 10
        }

        args.min_on = {
            'kettle': 2,
            'fridge': 10,
            'washing_machine': 300,
            'microwave': 2,
            'dishwasher': 300
        }

        args.min_off = {
            'kettle': 0,
            'fridge': 2,
            'washing_machine': 26,
            'microwave': 5,
            'dishwasher': 300
        }

        args.c0 = {
            'kettle': 1.,
            'fridge': 1e-6,
            'washing_machine': 0.01,
            'microwave': 1.,
            'dishwasher': 1.
        }

    args.optimizer = 'adam'
    args.lr = 1e-4
    args.enable_lr_schedule = False
    args.batch_size = 128


def acc_precision_recall_f1_score(pred, status):
    assert pred.shape == status.shape

    pred = pred.reshape(-1, pred.shape[-1])
    status = status.reshape(-1, status.shape[-1])
    accs, precisions, recalls, f1_scores = [], [], [], []

    for i in range(status.shape[-1]):
        tn, fp, fn, tp = confusion_matrix(status[:, i], pred[:, i], labels=[
                                          0, 1]).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / np.max((tp + fp, 1e-9))
        recall = tp / np.max((tp + fn, 1e-9))
        f1_score = 2 * (precision * recall) / \
            np.max((precision + recall, 1e-9))

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return np.array(accs), np.array(precisions), np.array(recalls), np.array(f1_scores)


def apply_min_on_off_constraints(status_pred, min_on, min_off):
    """
    Apply minimum on/off duration constraints to predicted appliance status.
    This post-processing step helps reduce rapid state switching and improves F1 scores.
    
    Args:
        status_pred: numpy array of predicted binary status (0/1)
        min_on: minimum duration (in samples) that appliance should stay on
        min_off: minimum duration (in samples) that appliance should stay off
    
    Returns:
        numpy array of filtered status predictions
    """
    if status_pred.ndim == 1:
        status_pred = status_pred.reshape(-1, 1)
    
    filtered_status = status_pred.copy()
    
    # Ensure min_on and min_off are arrays
    if not hasattr(min_on, '__getitem__'):
        min_on = np.array([min_on])
    if not hasattr(min_off, '__getitem__'):
        min_off = np.array([min_off])
    
    for appliance_idx in range(status_pred.shape[1]):
        status_seq = status_pred[:, appliance_idx].copy()
        
        # Handle indexing for single vs multiple appliances
        min_on_idx = min(appliance_idx, len(min_on) - 1)
        min_off_idx = min(appliance_idx, len(min_off) - 1)
        min_on_samples = int(min_on[min_on_idx])
        min_off_samples = int(min_off[min_off_idx])
        
        # Find state changes
        state_changes = np.diff(status_seq.astype(int))
        change_indices = np.where(state_changes != 0)[0] + 1
        
        if len(change_indices) == 0:
            continue
            
        # Add start and end points
        change_indices = np.concatenate([[0], change_indices, [len(status_seq)]])
        
        # Process each segment
        for i in range(len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]
            segment_length = end_idx - start_idx
            current_state = status_seq[start_idx]
            
            # Apply constraints
            if current_state == 1 and segment_length < min_on_samples:
                # ON period too short, set to OFF
                status_seq[start_idx:end_idx] = 0
            elif current_state == 0 and segment_length < min_off_samples:
                # OFF period too short, set to ON
                status_seq[start_idx:end_idx] = 1
        
        filtered_status[:, appliance_idx] = status_seq
    
    return filtered_status


def relative_absolute_error(pred, label):
    assert pred.shape == label.shape

    pred = pred.reshape(-1, pred.shape[-1])
    label = label.reshape(-1, label.shape[-1])
    temp = np.full(label.shape, 1e-9)
    relative, absolute, sum_err = [], [], []

    for i in range(label.shape[-1]):
        relative_error = np.mean(np.nan_to_num(np.abs(label[:, i] - pred[:, i]) / np.max(
            (label[:, i], pred[:, i], temp[:, i]), axis=0)))
        absolute_error = np.mean(np.abs(label[:, i] - pred[:, i]))

        relative.append(relative_error)
        absolute.append(absolute_error)

    return np.array(relative), np.array(absolute)
