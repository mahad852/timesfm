import numpy as np
import timesfm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

context_len = 512
pred_len = 64

# ecg_dataset = ECG_MIT(context_len=context_len, pred_len=pred_len, data_path="/home/user/MIT-BIH.npz")
data_path = "/home/mali2/datasets/ecg/MIT-BIH-splits.npz"

def single_loader(dataset):
    for i in range(1, len(dataset.files), 2):
        xy = dataset[dataset.files[i]]
        x, y = xy[:context_len], xy[context_len:context_len + pred_len]

        yield [x], [y]

dataset = np.load(data_path)


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

mse_by_pred_len = {}
rmse_by_pred_len = {}
mae_by_pred_len = {}

total = 0

for p_len in range(1, pred_len + 1):
    mse_by_pred_len[p_len] = 0.0
    rmse_by_pred_len[p_len] = 0.0
    mae_by_pred_len[p_len] = 0.0

for i, (x, y) in enumerate(single_loader(dataset)):
    point_forecast, experimental_quantile_forecast = tfm.forecast(x)

    point_forecast = np.array(point_forecast)
    y = np.array(y)

    mse = mean_squared_error(y, point_forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, point_forecast)

    mses.append(mse)
    maes.append(mae)

    total += 1

    for p_len in range(1, pred_len + 1):
        mse_by_pred_len[p_len] += mean_squared_error(y[:, :p_len], point_forecast[:, :p_len])
        mae_by_pred_len[p_len] += mean_absolute_error(y[:, :p_len], point_forecast[:, :p_len])

    if i % 20 == 0:
        print(f"iteraition: {i} | MSE: {mse} RMSE: {rmse} MAE: {mae}")

print(f"MSE: {np.average(mses)} RMSE: {np.sqrt(np.average(mses))} MAE: {np.average(maes)}")

for p_len in range(1, pred_len + 1):
    mse_by_pred_len[p_len] /= total
    rmse_by_pred_len[p_len] = np.sqrt(mse_by_pred_len[p_len])
    mae_by_pred_len[p_len] /= total

if not os.path.exists("logs"):
    os.mkdir("logs")

with open(os.path.join("logs", f"TimesFM_{context_len}_{pred_len}_split.csv"), "w") as f:
    f.write("context_len,horizon_len,MSE,RMSE,MAE\n")
    for p_len in range(1, pred_len + 1):
        f.write(f"{context_len},{p_len},{mse_by_pred_len[p_len]},{rmse_by_pred_len[p_len]},{mae_by_pred_len[p_len]}")
        if p_len != pred_len:
            f.write("\n")
