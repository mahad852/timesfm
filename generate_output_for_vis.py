import numpy as np
import timesfm
import os

dataset_path = "/Users/mali2/datasets/ecg/MIT-BIH_lagllama_384_64_forecast.npz"

sample_key = "a103-48_384_64_"
sample_true_key = f"{sample_key}true"

context_len = 384
pred_len = 64

def get_context(ds, true_key):
    return ds[true_key][:context_len]

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

dataset = np.load(dataset_path)
context = np.array([get_context(dataset, sample_true_key)])

point_forecast, experimental_quantile_forecast = tfm.forecast(context)
point_forecast = np.array(point_forecast)

if not os.path.exists("forecasts"):
    os.mkdir("forecasts")

np.save("forecasts/a103-48_384_64_lstm", point_forecast)