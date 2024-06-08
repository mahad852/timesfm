import numpy as np
from datasets.ecg_mit import ECG_MIT
import random
import timesfm
from sklearn.metrics import mean_squared_error, mean_absolute_error

context_len = 128
pred_len = 128
ecg_dataset = ECG_MIT(context_len=context_len, pred_len=pred_len, data_path="/home/user/MIT-BIH.npz")


max_len = 128
batch_size = 32

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
# print("Ecg dataset length:", len(ecg_dataset))

# print(ecg_dataset[1298433][0].shape,  ecg_dataset[1298433][1].shape)

# for i, (x, y) in enumerate(batch_loader(ecg_dataset, indices, batch_size)):
#     print(i, "|", len(x), len(y))

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

forecast_input = [
    np.sin(np.linspace(0, 20, 100)),
    np.sin(np.linspace(0, 20, 200)),
]
frequency_input = [0, 1, 2]


mses = []
maes = []
rmses = []

for i, (x, y) in enumerate(batch_loader(ecg_dataset, indices, batch_size)):
    point_forecast, experimental_quantile_forecast = tfm.forecast(
        x,
        # freq=frequency_input,
    )

    mse = mean_squared_error(np.array(y), np.array(point_forecast))
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(np.array(y), np.array(point_forecast))

    mses.append(mse)
    rmses.append(rmse)
    maes.append(mae)

# point_forecast, experimental_quantile_forecast = tfm.forecast(
#     forecast_input,
#     freq=frequency_input,
# )

# print(point_forecast, experimental_quantile_forecast)
# print(point_forecast[0].shape)

print(f"MSE: {np.average(mses)} RMSE: {np.average(rmses)} MAE: {np.average(maes)}")