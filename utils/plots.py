import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

key_dic = {
    'omega_X_1' : r'$\omega_{X,1}$',
    'omega_X_2' : r'$\omega_{X,2}$',
    'omega_T' : r'$\omega_T$'
}


def plot_treatment_probabilities_diagnostics(th_p_t, p_x, th_p_t_mx, p_xm, m,
                                             key, overlap_coefficient):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.kdeplot(th_p_t, color="green", ls='--', ax=axes[0],
                label='theoretical P(T=1|X)')
    sns.histplot(p_x, color="green", kde=True, stat='density', ax=axes[0],
                 label='estimated P(T=1|X)')
    # sns.kdeplot(1-th_p_t, color="green", ls='--', ax=axes[0], label='theoretical P(T=0|X)')
    # sns.histplot(1-p_x, color="green", kde=True, stat='density', ax=axes[0], label='estimated P(T=0|X)')

    axes[0].legend(bbox_to_anchor=[1, 0.9])

    ax = axes[1]
    sns.kdeplot(th_p_t_mx, color="orange", ls='--', ax=axes[1],
                label='theoretical P(T=1|X, M)')
    sns.kdeplot(1 - th_p_t_mx, color="red", ls='--', ax=axes[1],
                label='theoretical P(T=0|X, M)')
    # sns.kdeplot(th_p_t_mx[m==0], color="blue", ls='--', ax=axes[1], label='theoretical P(T=1|X, M=0)')
    sns.histplot(p_xm, color="orange", kde=True, bins=50, stat='density',
                 ax=axes[1], label='estimated P(T=1|X, M)')
    sns.histplot(1 - p_xm, color="red", kde=True, bins=50, stat='density',
                 ax=axes[1], label='estimated P(T=0|X, M)')
    # sns.histplot(p_xm[m==0], color="blue", kde=True, bins=50, stat='density', ax=axes[1], label='estimated P(T=1|X, M=0)')
    axes[1].legend(bbox_to_anchor=[1, 0.9])
    # sns.scatterplot(x=th_p_t_mx, y=p_xm, ax=ax)
    # ax.set_xticks(ax.get_yticks())
    # ax.set_aspect('equal', adjustable='box')
    # plt.plot([0, 1], [0, 1], 'red')
    # plt.ylabel('estimated P(T=1|X, M)')
    # plt.xlabel('theoretical P(T=1|X, M)')
    plt.subplots_adjust(wspace=1.4)
    fig.suptitle('Treatment probabilities - {} overlap {}'.format(key_dic[key],
                                                                  overlap_coefficient),
                 fontsize=20)
    plt.savefig('_treatment_probabilities_{}_overlap_{}.pdf'.format(key,
                                                                    overlap_coefficient))


def plot_mediator_density_diagnostics(mediator_density, p_m1, p_m0, key,
                                      overlap_coefficient):
    f_11x, f_10x, f_01x, f_00x = mediator_density

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    ax = axes[0]
    # sns.kdeplot(p_t, color="blue", ls='--', ax=axes[0], label='theoretical P(T=1|X)')
    sns.histplot(f_11x, color="blue", kde=True, stat='density', ax=axes[0],
                 label='estimated P(M=1|T=1,X)')
    sns.kdeplot(p_m1, color="blue", ls='--', ax=axes[0],
                label='theoretical P(M=1|T=1,X)')
    sns.histplot(f_10x, color="orange", kde=True, stat='density', ax=axes[0],
                 label='estimated P(M=0|T=1,X)')
    sns.kdeplot(1 - p_m1, color="orange", ls='--', ax=axes[0],
                label='theoretical P(M=0|T=1,X)')
    # sns.kdeplot(p_m_, color="red", ls='--', ax=axes[0], label='theoretical P(M|T,X)')

    # sns.kdeplot(th_p_t_mx, color="orange", ls='--', ax=axes[0], label='theoretical P(T=1|X, M)')
    # sns.histplot(p_xm, color="orange", kde=True, bins=50, stat='density', ax=axes[0], label='estimated P(T=1|X, M)')

    axes[0].legend(bbox_to_anchor=[1, 0.9])

    ax = axes[1]
    sns.histplot(f_01x, color="blue", kde=True, stat='density', ax=axes[1],
                 label='estimated P(M=1|T=0,X)')
    sns.kdeplot(p_m0, color="blue", ls='--', ax=axes[1],
                label='theoretical P(M=1|T=0,X)')
    sns.histplot(f_00x, color="orange", kde=True, stat='density', ax=axes[1],
                 label='estimated P(M=0|T=0,X)')
    sns.kdeplot(1 - p_m0, color="orange", ls='--', ax=axes[1],
                label='theoretical P(M=0|T=0,X)')

    axes[1].legend(bbox_to_anchor=[1, 0.9])

    plt.subplots_adjust(wspace=1.4)

    fig.suptitle('Mediator probabilities - {} overlap {}'.format(key_dic[key],
                                                                 overlap_coefficient),
                 fontsize=20)
    plt.savefig('_mediator_probabilities_{}_overlap_{}.pdf'.format(key,
                                                                   overlap_coefficient))
    

def visualize_errors(surfaces_total, surfaces_direct, surfaces_indirect):

    s_total = np.mean(surfaces_total, axis=0)
    s_direct = np.mean(surfaces_direct, axis=0)
    s_indirect = np.mean(surfaces_indirect, axis=0)

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(30, 8))

    # Plot each heatmap
    sns.heatmap(s_total, cmap='viridis', ax=axs[0], cbar_kws={'label': 'Error'})
    axs[0].set_title('Total effect')
    sns.heatmap(s_direct, cmap='viridis', ax=axs[1], cbar_kws={'label': 'Error'})
    axs[1].set_title('Direct effect')
    sns.heatmap(s_indirect, cmap='viridis', ax=axs[2], cbar_kws={'label': 'Error'})
    axs[2].set_title('Indirect effect')

    # Adjust layout to make room for the titles and colorbars
    plt.tight_layout()

    # Show the plot
    plt.show()

def visualize_treatment_curve(t):
    t_prime = np.random.permutation(t)
    plt.hist2d(t.squeeze(), t_prime.squeeze(), bins=40)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$t\prime$')
    plt.title('Treatment curve')
    plt.savefig('treatment_visualize.png')