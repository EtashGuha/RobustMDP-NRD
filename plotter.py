import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

    

def plot(obj_vals, name):
        fig, ax = plt.subplots()
        obj_vals = obj_vals
        sns.set_style("whitegrid", {
            "font.family": "serif",
            "font.serif": ["Times", "Palatino", "serif"]
        })
        sns.lineplot(
            ax=ax,
            x=list(range(len(obj_vals))),
            y=obj_vals,
            color='black',
            linewidth=2.8,
            marker='o',
            markerfacecolor='white',
            markeredgecolor='black'        
        )
        plt.ylabel(r'$\min_W \; V_W^{\pi}(\mu)$', fontsize=36)
        plt.xlabel(r'T', fontsize=36)
        plt.tight_layout()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.savefig("images/{}.png".format(name))

        fig.clf()