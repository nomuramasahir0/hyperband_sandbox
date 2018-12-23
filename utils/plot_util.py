import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def plot_separately(separate_dict, homedir):
    plt.figure()
    for k, v in separate_dict.items():
        val_len = len(v)
        vals = [v[i][1] for i in range(val_len)]
        plt.plot([i for i in range(val_len)], vals, markersize=20)
    plt.xlabel("epoch")
    plt.ylabel("validation error rate")
    plt.ylim(0.0, 10.0)
    plt.title("hyperband separate plot")
    plt.savefig(homedir + 'separate_plot.pdf')
