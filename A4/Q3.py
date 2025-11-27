
import numpy as np


def MDP_value_iteration(P, rewards,theta=1e-6):
    """
    Perform value iteration for a given MDP.

    Parameters:
    P : list of lists
        Transition probability matrix where P[s][s'] is the probability of transitioning from state s to state s'.
    rewards : list
        Reward for each state.
    gamma : float
        Discount factor.
    theta : float
        Convergence threshold.

    Returns:
    V : list
        The optimal value function.
    """
    V =rewards.copy()  # Initialize value function with immediate rewards
    k = 0
    while True:
   
        V_new = P @ V + rewards
        print(V_new,k)
        if np.linalg.norm(V_new - V) < theta:
            break
        V = V_new
        k += 1
    print("Converged in",k,"iterations.")
    return V




def main():
    P = np.array([[1,0,0,0,0,0,0,0],[0.5,0,0.5,0,0,0,0,0],[0,0.5,0,0.5,0,0,0,0],
         [0,0,0.5,0,0.5,0,0,0],[0,0,0,0.5,0,0.5,0,0],[0,0,0,0,0.5,0,0.5,0],
         [0,0,0,0,0,0.5,0,0.5],[0,0,0,0,0,0,0,1]])
    rewards = np.array([0,10,0,0,0,0,10,0])
    initial_v = rewards
    n_episodes = 1000
    optimal_values = MDP_value_iteration(P, rewards)
    print("Optimal Value Function:")    
    print(optimal_values)
    
    
if __name__ == "__main__":
    main()    