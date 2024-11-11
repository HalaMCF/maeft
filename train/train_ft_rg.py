import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
from maeft_data.math import math_train_data, math_val_data, math_test_data
from maeft_data.por import por_train_data, por_val_data, por_test_data
from sklearn.metrics import  mean_absolute_error
from utils.config import student_math, student_por
import numpy as np
import delu
import scipy
import math
from tqdm.std import tqdm
from torch import Tensor
from typing import Dict, Literal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = {"math": math_train_data, "por": por_train_data}
data_test = {"math": math_test_data, "por":por_test_data}
data_val = {"math": math_val_data, "por":por_val_data}
data_config = {"math": student_math, "por": student_por}

dataset = "insurance"
task_type = "regression"
this_config = data_config[dataset].input_bounds
if dataset == "math" or dataset == "por":
    n_c = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
elif dataset == "insurance":
    n_c = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]

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
    lr=1e-3,
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

if task_type != "multiclass":
    # Required by F.binary_cross_entropy_with_logits
    for part in data:
        data[part]["y"] = data[part]["y"].float()


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
    else F.l1_loss
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

    if task_type == "regression":
        assert task_type == "regression"
        score = mean_absolute_error(y_true, y_pred)
    
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
early_stopping = delu.tools.EarlyStopping(patience, mode="min")
best = {
    "val": 100,
    "test": 100,
    "epoch": -1,
}


timer.run()
for epoch in range(n_epochs):
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

    if val_score < best["val"]:
        print("🌸 New best epoch! 🌸")
        best = {"val": val_score, "test": test_score, "epoch": epoch}
        torch.save(model.state_dict(), '{}_ft.pth'.format(dataset))
    print()

print("\n\nResult:")
print(best)  