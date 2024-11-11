import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
from maeft_data.census import census_train_data, census_val_data, census_test_data
from maeft_data.credit import credit_train_data, credit_val_data, credit_test_data
from maeft_data.bank import bank_train_data, bank_val_data, bank_test_data
from maeft_data.meps import meps_train_data, meps_val_data, meps_test_data
from maeft_data.tae import tae_train_data, tae_val_data, tae_test_data
from maeft_data.ricci import ricci_train_data, ricci_val_data, ricci_test_data
from maeft_data.compas import compas_train_data, compas_val_data, compas_test_data
from sklearn.metrics import accuracy_score
from utils.config import census, credit, bank, meps, ricci, tae, compas
import numpy as np
import delu
import scipy
import math
from tqdm.std import tqdm
from torch import Tensor
from typing import Dict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = {"census":census_train_data, "credit": credit_train_data, "bank": bank_train_data, "meps": meps_train_data, "tae": tae_train_data, "ricci": ricci_train_data, "compas": compas_train_data}
data_test = {"census":census_test_data, "credit": credit_test_data, "bank": bank_test_data, "meps": meps_test_data, "tae": tae_test_data, "ricci": ricci_test_data, "compas": compas_test_data}
data_val = {"census":census_val_data, "credit": credit_val_data, "bank": bank_val_data, "meps":meps_val_data, "tae": tae_val_data, "ricci": ricci_val_data, "compas": compas_val_data}
data_config = {"census": census, "credit": credit, "bank": bank, "meps": meps, "ricci": ricci, "tae": tae, "compas": compas}

dataset = "ricci"
task_type = "binclass"
this_config = data_config[dataset].input_bounds

if dataset == "census":
    n_c = [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
elif dataset == "bank":
    n_c = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
elif dataset == "meps":
    n_c = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
elif dataset == "credit":
    n_c = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
elif dataset == "ricci" or dataset == "tae":
    n_c = [0, 1, 1, 0, 1]
elif dataset == "compas":
    n_c = [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]


X_train, Y_train, input_shape, nb_classes = data[dataset]()


d_out = 1 
x_cont = []
x_cat = []
for i in X_train:
    temp_cont = []
    temp_cat = []
    for j in range(len(i)):
        if n_c[j] == 1:
            temp_cont.append(i[j])
        else:
            temp_cat.append(i[j])
    x_cont.append(temp_cont)
    x_cat.append(temp_cat)
x_cont = torch.Tensor(x_cont).to(torch.int64).to(device)
x_cat = torch.Tensor(x_cat).to(torch.int64).to(device)
n_cont_features = len(x_cont[0])
n_cat_features = len(x_cat[0])
cat_cardinalities = []
for i in range(len(this_config)):
    if n_c[i] == 0:
        temp = this_config[i][1] + 1
        cat_cardinalities.append(temp)
        
X_test, Y_test, input_shape, nb_classes = data_test[dataset]()
x_cont_test = []
x_cat_test = []
for i in X_test:
    temp_cont = []
    temp_cat = []
    for j in range(len(i)):
        if n_c[j] == 1:
            temp_cont.append(i[j])
        else:
            temp_cat.append(i[j])
    x_cont_test.append(temp_cont)
    x_cat_test.append(temp_cat)
x_cont_test = torch.Tensor(x_cont_test).to(torch.int64).to(device)
x_cat_test = torch.Tensor(x_cat_test).to(torch.int64).to(device)


X_valid, Y_valid, input_shape, nb_classes = data_val[dataset]()
x_cont_valid = []
x_cat_valid = []
for i in X_valid:
    temp_cont = []
    temp_cat = []
    for j in range(len(i)):
        if n_c[j] == 1:
            temp_cont.append(i[j])
        else:
            temp_cat.append(i[j])
    x_cont_valid.append(temp_cont)
    x_cat_valid.append(temp_cat)
x_cont_valid = torch.Tensor(x_cont_valid).to(torch.int64).to(device)
x_cat_valid = torch.Tensor(x_cat_valid).to(torch.int64).to(device)

#prepare FTTransformer as tutorial
model = FTTransformer(
    n_cont_features=n_cont_features,
    cat_cardinalities=cat_cardinalities,
    d_out=d_out,
    n_blocks=3,
    d_block=192,
    attention_n_heads=8,
    attention_dropout=0.2,
    ffn_d_hidden=None,
    ffn_d_hidden_multiplier=4 / 3,
    ffn_dropout=0.1,
    residual_dropout=0.0,
).to(device)
optimizer = torch.optim.AdamW(
    model.make_parameter_groups(),
    lr=1e-4,
    weight_decay=1e-5,
) 
data_numpy = {
    "train": {"x_cont": x_cont, "x_cat": x_cat,"y": Y_train},
    "val": {"x_cont": x_cont_valid, "x_cat": x_cat_valid, "y": Y_valid},
    "test": {"x_cont": x_cont_test, "x_cat": x_cat_test, "y": Y_test},
}
data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}


def apply_model(batch: Dict[str, Tensor]) -> Tensor:
    if isinstance(model, FTTransformer):
        return model(batch["x_cont"], batch.get("x_cat")).squeeze(-1)

    else:
        raise RuntimeError(f"Unknown model type: {type(model)}")


loss_fn = (
    F.binary_cross_entropy_with_logits
    if task_type == "binclass"
    else F.cross_entropy
    if task_type == "multiclass"
    else F.mse_loss
)


@torch.no_grad()
def evaluate(part: str) -> float:
    model.eval()

    eval_batch_size = 8096
    y_pred = (
        torch.cat(
            [
                apply_model(batch)
                for batch in delu.iter_batches(data[part], eval_batch_size)
            ]
        )
        .cpu()
        .numpy()
    )
    y_true = data[part]["y"].cpu().numpy()

    if task_type == "binclass":
        y_pred = np.round(scipy.special.expit(y_pred))
        #print(y_pred)
        score = accuracy_score(y_true, y_pred)
    
    return score  # The higher -- the better.


# For demonstration purposes (fast training and bad performance),
# one can set smaller values:
# n_epochs = 20
# patience = 2
n_epochs = 300
patience = 40

batch_size = 512
epoch_size = math.ceil(len(X_train) / batch_size)
timer = delu.tools.Timer()
early_stopping = delu.tools.EarlyStopping(patience, mode="max")
best = {
    "val": -math.inf,
    "test": -math.inf,
    "epoch": -1,
}


timer.run()
for epoch in range(1):
    for batch in tqdm(
        delu.iter_batches(data["train"], batch_size, shuffle=True),
        desc=f"Epoch {epoch}",
        total=epoch_size,
    ):  
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(apply_model(batch), batch["y"])
        loss.backward()
        optimizer.step()

    val_score = evaluate("val")
    test_score = evaluate("test")
    print(f"(val) {val_score:.4f} (test) {test_score:.4f} [time] {timer}")

    early_stopping.update(val_score)
    if early_stopping.should_stop():
        break

    if val_score > best["val"]:
        print("🌸 New best epoch! 🌸")
        best = {"val": val_score, "test": test_score, "epoch": epoch}
        torch.save(model.state_dict(), '{}_ft.pth'.format(dataset))
    print()

print("\n\nResult:")
print(best)  