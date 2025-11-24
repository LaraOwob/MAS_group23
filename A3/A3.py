


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

def Thompson_update(alpha, beta, reward):
    """
    Perform a Thompson sampling update for a Bernoulli bandit.

    Parameters:
    alpha (int): The number of successes (reward = 1).
    beta (int): The number of failures (reward = 0).
    reward (int): The observed reward (0 or 1).

    Returns:
    tuple: Updated (alpha, beta) values.
    """
    alpha =alpha + reward
    beta = beta +(1-reward)
    return alpha, beta
def beta_density_plot(iterations, p_list):

    alpha_beta_pairs = []
    for p in p_list:
        alpha, beta = 1, 1  # Reset prior for each p
        rewards = create_reward(p, iterations)

        for reward in rewards:
            alpha, beta = Thompson_update(alpha, beta, reward)
            alpha_beta_pairs.append((alpha, beta))

        print(f'For true p={p}, updated alpha: {alpha}, beta: {beta}')

        x = np.linspace(0, 1, 500)   # smoother curve
        y = beta_dist.pdf(x, alpha, beta)

        plt.plot(x, y, label=f'Beta({alpha}, {beta})  p={p}')

    plt.title('Beta Distribution')
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.grid()

    # Legend OUTSIDE the plot (right side)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()  # adjusts layout to prevent clipping

    plt.savefig("A3_beta_distributions.png", bbox_inches='tight')
    plt.show()

    
def create_reward(p,iterations):
    reward_list = []
    for _ in range(iterations):
        if np.random.rand() < p:
            reward_list.append(1)
        else:
            reward_list.append(0)
            
    return reward_list


def main():
    list_p = [i/10 for i in range(1,10)]
    no_iterations = 1000
    beta_density_plot(no_iterations,list_p)


if __name__ == "__main__":
    main()
