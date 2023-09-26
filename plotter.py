import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def simple_moving_average(data):
    """
    Calculate the simple moving average of a list of data points.
    
    Args:
    data (list): The input data to be smoothed.
    window_size (int): The size of the moving average window.
    
    Returns:
    list: A list of smoothed values.
    """
    smoothed_data = []
    for i in range(1, len(data) + 1):
        smoothed_data.append(np.max(data[:i]) - np.std(data[:i])/np.sqrt(i))
    return smoothed_data

    

def plot(obj_vals, name):
        fig, ax = plt.subplots()
        obj_vals = obj_vals[:40]
        sns.set_style("whitegrid", {
            "font.family": "serif",
            "font.serif": ["Times", "Palatino", "serif"]
        })
        sns.lineplot(
            ax=ax,
            x=list(range(len(obj_vals))),
            y=simple_moving_average(obj_vals),
            color='black',
            linewidth=2.8,
            marker='o',
            markerfacecolor='white',
            markeredgecolor='black'        
        )
        ax.set(ylabel=r'$\min_W V_W^{\pi}(\mu)$', xlabel=r'T')
        plt.tight_layout()

        # if args.dataset_name == "bimodal" or args.dataset_name == "log_normal":
        #     _, y, _, _ = get_train_val_data(args)
        #     hist, bins = np.histogram(y, bins=args.range_size)
        #     # Calculate bin centers
        #     bin_centers = (bins[:-1] + bins[1:]) / 2
        #     plt.plot(bin_centers, hist/len(y), label="true distribution")

        # percentile_val = percentile_excluding_index(all_scores, alpha)
        # coverage, length = calc_length_coverage(scores[i], range_vals, percentile_val, y_val[i])
        # ax.axhline(y=percentile_val.detach().numpy(), label=r'Confidence Level $\alpha$', color='#a8acb3', linestyle='--',)
        # ax.axvline(x=y_val[i].detach().numpy(), label=r'Ground Truth $y_{n+1}$', color='#646566', linestyle=':',)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.savefig("images/{}.png".format(name))
        # ax.legend()
        # if coverage == 1:
        #     fig.savefig("images/{}/right/{}.png".format(args.model_path, i))
        # else:
        #     fig.savefig("images/{}/wrong/{}.png".format(args.model_path, i))
        fig.clf()