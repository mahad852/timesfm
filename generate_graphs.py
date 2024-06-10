import matplotlib.pyplot as plt
import os
import pandas as pd

if not os.path.exists("graphs"):
    os.mkdir("graphs")

mses_by_context_len = {}
rmses_by_context_len = {}
maes_by_context_len = {}
pred_lens = [p_len for p_len in range(1, 65, 1)]


def plot_graph(x, y, path, title, x_label, y_label):
    fig = plt.figure()
    ax = fig.add_subplot()

    for k in y.keys():
        ax.plot(x, y[k], label=f"Context {k}")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.legend(loc='best')
    
    fig.savefig(path)

for fname in os.listdir("logs"):
    if fname.split('.')[-1] != "csv":
        continue

    data = pd.read_csv(os.path.join("logs", fname))

    context_len = data["context_len"][0]

    mses_by_context_len[context_len] = data["MSE"]
    rmses_by_context_len[context_len] = data["RMSE"]
    maes_by_context_len[context_len] = data["MAE"]


plot_graph(pred_lens, mses_by_context_len, "graphs/mse.png", "MSE v.s Forecast Horizon", "Forecast Horizon", "Mean Squared Error (MSE)")
plot_graph(pred_lens, rmses_by_context_len, "graphs/rmse.png", "RMSE v.s Forecast Horizon", "Forecast Horizon", "Root Mean Squared Error (RMSE)")
plot_graph(pred_lens, maes_by_context_len, "graphs/mae.png", "MAE v.s Forecast Horizon", "Forecast Horizon", "Mean Absolute Error (MAE)")