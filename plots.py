from matplotlib import pyplot as plt
import numpy as np
import json
import os

DATASET = "PEMS08"


# calculate mean and std for each metric at each horizon
metrics = ["mae", "rmse", "mape"]


def plot_metrics_for_different_runs():
    data = []
    analyzed_data = {metric: {} for metric in metrics}

    # read `test_metrics-s*.json` files
    for i in range(1, 7):
        with open(f"pretrained-model/{DATASET}/test_metrics-s{i}.json") as f:
            data.append(json.load(f))

    for metric in metrics:
        for h in range(12):
            mean = np.mean([data[i][h][metric] for i in range(len(data))], axis=0)
            std = np.std([data[i][h][metric] for i in range(len(data))], axis=0)
            analyzed_data[metric][h] = (mean, std)

    os.makedirs("plots", exist_ok=True)

    # plot
    for metric in metrics:
        plt.figure()
        # plt.title(metric.upper() + " for Dataset " + DATASET + " with confidence interval (95%)")
        for h in range(12):
            ## draw confidence interval
            confidence = 1.96 * analyzed_data[metric][h][1] / np.sqrt(len(data))
            plt.errorbar(
                h + 1, analyzed_data[metric][h][0], yerr=confidence, c="b", marker="o"
            )
        plt.xticks(np.arange(1, 13, 1.0))
        plt.grid(True)

        plt.plot(
            np.arange(1, 13, 1.0),
            [analyzed_data[metric][h][0] for h in range(12)],
            c="gray",
        )

        plt.xlabel("horizon")
        plt.ylabel(metric.upper())
        plt.savefig(f"plots/{DATASET}-{metric}.png")


def plot_embedding_dim_metrics():
    data = []
    dims = [2, 5, 10, 15, 20]

    # read `test_metrics-s4-dim*.json` files
    for i in dims:
        with open(f"pretrained-model/{DATASET}/test_metrics-s4-dim{i}.json") as f:
            data.append(json.load(f))

    fig, ax = plt.subplots()
    
    ax.plot(dims, [data[d][12][metrics[0]] for d in range(5)], color="red", marker="o")
    ax.set_xlabel("Embedding Dimension", fontsize = 14)
    ax.set_xticks(dims)
    ax.set_ylabel(metrics[0].upper(), color="red", fontsize=14)

    ## another y-axis on the right
    ax2=ax.twinx()
    ax2.plot(dims, [data[d][12][metrics[2]] for d in range(5)], color="blue", marker="o")
    ax2.set_ylabel(metrics[2].upper(), color="blue", fontsize=14)
    
    # save the plot as a file
    fig.savefig(f"plots/{DATASET}-dim.png")


def print_table_data():
    data = []
    for i in range(1, 7):
        with open(f"pretrained-model/{DATASET}/test_metrics-s{i}.json") as f:
            data.append(json.load(f))

    # calculate mean and std for each metric at each horizon
    metrics = ["mae", "rmse", "mape"]
    average_horizon_analyzed = {metric: {} for metric in metrics}

    for metric in metrics:
        # average over all 12 horizons is the last one
        mean = np.mean([data[i][12][metric] for i in range(len(data))], axis=0)
        std = np.std([data[i][12][metric] for i in range(len(data))], axis=0)
        average_horizon_analyzed[metric]["mean"] = mean
        average_horizon_analyzed[metric]["std"] = std
        average_horizon_analyzed[metric]["confidence"] = 1.96 * std / np.sqrt(len(data))

    print(json.dumps(average_horizon_analyzed, indent=4))


if __name__ == "__main__":
    plot_metrics_for_different_runs()
    # print_table_data()
    plot_embedding_dim_metrics()
