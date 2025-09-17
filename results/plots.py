import matplotlib.pyplot as plt
import numpy as np

def plot_grouped_stacked(groups, group_labels=("Group1", "Group2"), bar_labels=None):
    """
    Plot grouped stacked bars for two groups of dicts.
    
    Args:
        groups: list of 2 groups, each a list of 3 dicts with keys
                'embedding', 'attention', 'fnn'
        group_labels: names for the two groups
        bar_labels: labels for the 3 bars (default: Bar1, Bar2, Bar3)
    """
    assert len(groups) == 2, "Must provide exactly 2 groups"
    assert all(len(g) == 3 for g in groups), "Each group must have 3 dicts"

    if bar_labels is None:
        bar_labels = [f"Bar{i+1}" for i in range(3)]

    categories = ["embedding", "attention", "fnn"]
    colors = {"embedding": "skyblue", "attention": "orange", "fnn": "green"}

    n_bars = 3
    group_spacing = 1.0      # distance between bar centers in a group
    bar_width = 0.35         # width of each stacked bar

    # x positions: 3 groups (bar1, bar2, bar3), each with 2 bars side-by-side
    x = np.arange(n_bars)

    fig, ax = plt.subplots(figsize=(8, 5))

    for g_idx, group in enumerate(groups):
        offsets = (-0.5, 0.5)  # left/right offset
        pos = x + offsets[g_idx] * bar_width

        for i, cat in enumerate(categories):
            bottoms = np.sum([[d[c] for c in categories[:i]] for d in group], axis=1)
            heights = [d[cat] for d in group]
            ax.bar(pos, heights, bottom=bottoms, color=colors[cat],
                   width=bar_width, label=cat if g_idx==0 and i==0 else "")

        # Add group labels under each bar
        for xi, label in zip(pos, [group_labels[g_idx]]*n_bars):
            ax.annotate(label, xy=(xi, 0), xytext=(0, -18),
                        textcoords="offset points", ha="center", va="top", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("Latency (S)")
    ax.set_title("Grouped Latency Breakdown")

    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate legend entries
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", frameon=False, ncols=3)

    plt.tight_layout()
    plt.savefig("tmp.png")