import itertools
import scipy.special
import torch
import numpy as np
import scipy

def mlp_check_for_error_condition(model, t, sens, low_bound, high_bound, k, n_c, device, task_type):
    combinations = list(itertools.product(*[range(low_bound[i], high_bound[i]) for i in range(len(sens))]))
    to_check = np.tile(t, (len(combinations), 1))
    for idx, sens_attr in enumerate(sens):
        column_values = [combo[idx] for combo in combinations]
        to_check[:, sens_attr] = column_values
    to_check_tensor = torch.tensor(to_check, dtype=torch.float).to(device)
    with torch.no_grad():
        result = model(to_check_tensor)
    result = result.argmax(dim=1, keepdim=True).cpu().numpy()
    return len(np.unique(result)) != 1

def mlp_check_for_error_condition_rg(model, t, sens, low_bound, high_bound, k, n_c, device, task_type):
    combinations = list(itertools.product(*[range(low_bound[i], high_bound[i]) for i in range(len(sens))]))
    to_check = np.tile(t, (len(combinations), 1))
    for idx, sens_attr in enumerate(sens):
        column_values = [combo[idx] for combo in combinations]
        to_check[:, sens_attr] = column_values
    to_check_tensor = torch.tensor(to_check, dtype=torch.float).to(device)
    with torch.no_grad():
        distances_input = torch.cdist(to_check_tensor, to_check_tensor, p=2)
        result = model(to_check_tensor)
        distances_output = torch.cdist(result, result, p=2)
        fairness_check = distances_output <= k * distances_input
    return len(torch.unique(fairness_check.cpu())) != 1


def ml_check_for_error_condition(model, t, sens, low_bound, high_bound, k, n_c, device, task_type):
    combinations = list(itertools.product(*[range(low_bound[i], high_bound[i]) for i in range(len(sens))]))
    to_check = np.tile(t, (len(combinations), 1))
    for idx, sens_attr in enumerate(sens):
        column_values = [combo[idx] for combo in combinations]
        to_check[:, sens_attr] = column_values
    to_check = to_check.astype(int)
    result = model.predict(np.vstack(to_check))
    return len(np.unique(result)) != 1


def ml_check_for_error_condition_rg(model, t, sens, low_bound, high_bound, k, n_c, device, task_type):
    combinations = list(itertools.product(*[range(low_bound[i], high_bound[i]) for i in range(len(sens))]))
    to_check = np.tile(t, (len(combinations), 1))
    for idx, sens_attr in enumerate(sens):
        column_values = [combo[idx] for combo in combinations]
        to_check[:, sens_attr] = column_values
    to_check_tensor = torch.tensor(to_check, dtype=torch.float).to(device)
    with torch.no_grad():
        predictions = model.predict(to_check.astype(int))
    predictions_tensor = torch.tensor(predictions, dtype=torch.float).to(device)
    distances_input = torch.cdist(to_check_tensor, to_check_tensor, p=2)
    distances_output = torch.cdist(predictions_tensor.unsqueeze(1), predictions_tensor.unsqueeze(1), p=2)
    fairness_check = distances_output <= k * distances_input
    return len(torch.unique(fairness_check.cpu())) != 1


def ft_check_for_error_condition(model, t, sens, low_bound, high_bound, k, n_c, device, task_type):
    combinations = list(itertools.product(*[range(low_bound[i], high_bound[i]) for i in range(len(sens))]))
    to_check = np.tile(t, (len(combinations), 1))
    for idx, sens_attr in enumerate(sens):
        column_values = [combo[idx] for combo in combinations]
        to_check[:, sens_attr] = column_values
    cont_indices = [j for j in range(len(n_c)) if n_c[j] == 1]
    cat_indices = [j for j in range(len(n_c)) if n_c[j] != 1]
    this_cont = to_check[:, cont_indices]
    this_cat = to_check[:, cat_indices]
    this_cont_tensor = torch.tensor(this_cont, dtype=torch.int64).to(device)
    this_cat_tensor = torch.tensor(this_cat, dtype=torch.int64).to(device)
    with torch.no_grad():
        result_tensor = model(this_cont_tensor, this_cat_tensor)
    result = result_tensor.detach().cpu().numpy()
    if task_type == 'binclass':
        result = np.round(scipy.special.expit(result))
    else:
        result = np.argmax(result, axis=1)
    return len(np.unique(result)) != 1


def ft_check_for_error_condition_rg(model, t, sens, low_bound, high_bound, k, n_c, device, task_type):
    combinations = list(itertools.product(*[range(low_bound[i], high_bound[i]) for i in range(len(sens))]))
    to_check = np.tile(t, (len(combinations), 1))
    for idx, sens_attr in enumerate(sens):
        column_values = [combo[idx] for combo in combinations]
        to_check[:, sens_attr] = column_values
    to_check_tensor = torch.tensor(np.vstack(to_check), dtype=torch.float).to(device)
    cont_indices = [j for j in range(len(n_c)) if n_c[j] == 1]
    cat_indices = [j for j in range(len(n_c)) if n_c[j] != 1]
    this_cont = to_check[:, cont_indices]
    this_cat = to_check[:, cat_indices]
    this_cont_tensor = torch.tensor(this_cont, dtype=torch.int64).to(device)
    this_cat_tensor = torch.tensor(this_cat, dtype=torch.int64).to(device)
    with torch.no_grad():
        distances_input = torch.cdist(to_check_tensor, to_check_tensor, p=2)
        result = model(this_cont_tensor, this_cat_tensor)
        distances_output = torch.cdist(result, result, p=2)
        fairness_check = distances_output <= k * distances_input
    return len(torch.unique(fairness_check.cpu())) != 1 