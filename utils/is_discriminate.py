import itertools
import torch
import numpy as np
import scipy

def mlp_check_for_error_condition(model, t, sens, low_bound, high_bound, k, n_c, device):
    combinations = list(itertools.product(*[range(low_bound[i], high_bound[i]) for i in range(len(sens))]))
    to_check = []
    for combo in combinations:
        temp = t.copy()
        for idx, sens_attr in enumerate(sens):
            temp[sens_attr] = combo[idx]
        to_check.append(temp)
    result = model(torch.tensor(np.vstack(to_check), dtype=torch.float).to(device))
    result = result.argmax(dim=1, keepdim=True).cpu()
    if len(np.unique(result)) != 1:
        return True
    return False
    
def mlp_check_for_error_condition_rg(model, t, sens, low_bound, high_bound, k, n_c, device):
    combinations = list(itertools.product(*[range(low_bound[i], high_bound[i]) for i in range(len(sens))]))
    to_check = []
    for combo in combinations:
        temp = t.copy()
        for idx, sens_attr in enumerate(sens):
            temp[sens_attr] = combo[idx]
        to_check.append(temp)
    to_check = torch.tensor(np.vstack(to_check), dtype=torch.float).to(device)
    distances_input = torch.cdist(to_check, to_check, p=2)
    distances_output = torch.cdist(model(to_check), model(to_check), p=2)
    fairness_check = distances_output <= k * distances_input
    if len(np.unique(fairness_check.cpu())) != 1:
        return True
    return False

def ml_check_for_error_condition(model, t, sens, low_bound, high_bound, k, n_c, device):
    combinations = list(itertools.product(*[range(low_bound[i], high_bound[i]) for i in range(len(sens))]))
    result = []
    for combo in combinations:
        temp = t.copy()
        for idx, sens_attr in enumerate(sens):
            temp[sens_attr] = combo[idx]
        result.append(model.predict(np.array(temp).astype(int)))
    result = np.array(result).astype(float)
    if len(np.unique(result)) != 1:
        return True
    return False

def ml_check_for_error_condition_rg(model, t, sens, low_bound, high_bound, k, n_c, device):
    combinations = list(itertools.product(*[range(low_bound[i], high_bound[i]) for i in range(len(sens))]))
    to_check = []
    res = []
    for combo in combinations:
        temp = t.copy()
        for idx, sens_attr in enumerate(sens):
            temp[sens_attr] = combo[idx]
        to_check.append(temp)
        res.append([model.predict(np.array(temp).astype(int))])
    to_check = np.array(to_check).astype(int)
    res = np.array(res).astype(float)
    distances_input = torch.cdist(torch.tensor(to_check, dtype=torch.float), torch.tensor(to_check, dtype=torch.float), p=2)
    distances_output = torch.cdist(torch.tensor(res, dtype=torch.float), torch.tensor(res, dtype=torch.float), p=2)
    fairness_check = distances_output <= k * distances_input

    if len(np.unique(fairness_check.cpu())) != 1:
        return True
    return False

def ft_check_for_error_condition(model, t, sens, low_bound, high_bound, k, n_c, device):
    combinations = list(itertools.product(*[range(low_bound[i], high_bound[i]) for i in range(len(sens))]))
    to_check = []
    for combo in combinations:
        temp = t.copy()
        for idx, sens_attr in enumerate(sens):
            temp[sens_attr] = combo[idx]
        to_check.append(temp)
    this_cont = []
    this_cat = []
    for i in to_check:
        temp_cont = []
        temp_cat = []
        for j in range(len(i)):
            if n_c[j] == 1:
                temp_cont.append(i[j])
            else:
                temp_cat.append(i[j])
        this_cont.append(temp_cont)
        this_cat.append(temp_cat)
    this_cont = torch.Tensor(this_cont).to(torch.int64).to(device)
    this_cat = torch.Tensor(this_cat).to(torch.int64).to(device)
    result = model(this_cont, this_cat).detach().cpu().numpy()
    result = np.round(scipy.special.expit(result))
    if len(np.unique(result)) != 1:
        return True
    return False

def ft_check_for_error_condition_rg(model, t, sens, low_bound, high_bound, k, n_c, device):
    combinations = list(itertools.product(*[range(low_bound[i], high_bound[i]) for i in range(len(sens))]))
    to_check = []
    for combo in combinations:
        temp = t.copy()
        for idx, sens_attr in enumerate(sens):
            temp[sens_attr] = combo[idx]
        to_check.append(temp)
    this_cont = []
    this_cat = []
    for i in to_check:
        temp_cont = []
        temp_cat = []
        for j in range(len(i)):
            if n_c[j] == 1:
                temp_cont.append(i[j])
            else:
                temp_cat.append(i[j])
        this_cont.append(temp_cont)
        this_cat.append(temp_cat)
    to_check = torch.tensor(np.vstack(to_check), dtype=torch.float).to(device)
    this_cont = torch.Tensor(this_cont).to(torch.int64).to(device)
    this_cat = torch.Tensor(this_cat).to(torch.int64).to(device)
    distances_input = torch.cdist(to_check, to_check, p=2)
    distances_output = torch.cdist(model(this_cont, this_cat), model(this_cont, this_cat), p=2)
    fairness_check = distances_output <= k * distances_input
    if len(np.unique(fairness_check.cpu())) != 1:
        return True
    return False