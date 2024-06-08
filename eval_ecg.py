import numpy as np
from datasets.ecg_mit import ECG_MIT
import random
import timesfm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

context_len = 512
pred_len = 64
ecg_dataset = ECG_MIT(context_len=context_len, pred_len=pred_len, data_path="/home/user/MIT-BIH.npz")


max_len = 260000
batch_size = 64

def single_loader(dataset: ECG_MIT, indices: list[int]):
    for index in indices:
        x, y = dataset[index]
        yield [x], [y]

def batch_loader(dataset: ECG_MIT, indices: list[int], batch_size: int):
    for i in range(0, len(indices), batch_size):
        xs = []
        ys = []

        for index in indices[i : min(i + batch_size, len(indices))]:
            x, y = dataset[index]
            xs.append(x)
            ys.append(y)

        yield xs, ys


indices = random.sample(range(len(ecg_dataset)), max_len)

tfm = timesfm.TimesFm(
    context_len=context_len,
    horizon_len=pred_len,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend="gpu",
)
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

mses = []
maes = []
rmses = []

mse_by_pred_len = {}
rmse_by_pred_len = {}
mae_by_pred_len = {}

total = 0

for p_len in range(1, pred_len + 1):
    mse_by_pred_len[p_len] = 0.0
    rmse_by_pred_len[p_len] = 0.0
    mae_by_pred_len[p_len] = 0.0

for i, (x, y) in enumerate(batch_loader(ecg_dataset, indices, batch_size)):
    point_forecast, experimental_quantile_forecast = tfm.forecast(x)

    point_forecast = np.array(point_forecast)
    y = np.array(y)

    mse = mean_squared_error(y, point_forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, point_forecast)

    mses.append(mse)
    rmses.append(rmse)
    maes.append(mae)

    total += 1

    for p_len in range(1, pred_len + 1):
        mse_by_pred_len[p_len] += mean_squared_error(y[:, :p_len], point_forecast[:, :p_len])
        rmse_by_pred_len[p_len] += np.sqrt(mean_squared_error(y[:, :p_len], point_forecast[:, :p_len]))
        mae_by_pred_len[p_len] += mean_absolute_error(y[:, :p_len], point_forecast[:, :p_len])

    if i % 20 == 0:
        print(f"iteraition: {i} | MSE: {mse} RMSE: {rmse} MAE: {mae}")

print(f"MSE: {np.average(mses)} RMSE: {np.average(rmses)} MAE: {np.average(maes)}")

for p_len in range(1, pred_len + 1):
    mse_by_pred_len[p_len] /= total
    rmse_by_pred_len[p_len] /= total
    mae_by_pred_len[p_len] /= total

if not os.path.exists("logs"):
    os.mkdir("logs")

with open(os.path.join("logs", f"TimesFM_{context_len}_{pred_len}.csv"), "w") as f:
    f.write("context_len,horizon_len,MSE,RMSE,MAE\n")
    for p_len in range(1, pred_len + 1):
        f.write(f"{context_len},{p_len},{mse_by_pred_len[p_len]},{rmse_by_pred_len[p_len]},{mae_by_pred_len[p_len]}")
        if p_len != pred_len:
            f.write("\n")