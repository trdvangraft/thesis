import seaborn as sns
import matplotlib.pyplot as plt


def make_strategies_plot(df, ax, plot_type: str = 'boxplot'):
    shared_plot_args = dict(
        data=df, x='strategy', y=PLOT_KWARGS[METRIC]['y'],
        palette='Greys', hue='strategy', dodge=False, ax=ax
    )

    if plot_type == 'boxplot':
        g0 = sns.boxplot(
            **shared_plot_args,
            width=.4,
        )
    elif plot_type == 'violin':
        g0 = sns.violinplot(
            **shared_plot_args,
            inner='point',
        )
    g0.set(
        xticklabels=[],
        xlabel="Strategies",
        **SET_KWARGS[METRIC]
    )
    g0.tick_params(bottom=False)
    ax.set_title("A", loc="left")
    g0.get_legend().remove()
    return g0

def make_architecture_plot(df, ax):
    g1 = sns.boxplot(
        data=df,
        x='architecture',
        y=PLOT_KWARGS[METRIC]['y'],
        palette=PLOT_KWARGS[METRIC]['palette'],
        ax=ax,
        width=.4,
    )
    g1.set(
        xticklabels=[],
        xlabel="Architectures",
        **SET_KWARGS[METRIC]
    )
    ax.set_title("B", loc="left")
    g1.tick_params(bottom=False)
    # g1.get_legend().remove()
    return g1

def make_metabolite_plot(df, ax):
    g2 = sns.barplot(
        data=df,
        hue='architecture',
        **PLOT_KWARGS[METRIC],
        ax=ax,
    )

    g2.set_xticklabels(g2.axes.get_xticklabels(), rotation=45)
    g2.set(
        xlabel='Metabolite id',
        **SET_KWARGS[METRIC]
    )
    ax.set_title("C", loc='left')
    g2.get_legend().remove()
    return g2

def make_legend(ax0, ax2):
    handles, labels = ax0.get_legend_handles_labels()
    leg0 = plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1.02), title='Strategy', alignment='left')
    ax2.add_artist(leg0)

    handles, labels = ax2.get_legend_handles_labels()
    leg1 = plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 0.75), title='Architecture', alignment='left')
    ax2.add_artist(leg1)
    return (leg0, leg1)