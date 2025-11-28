
import numpy as np


def MDP_value_iteration(P, rewards,theta=1e-6):

    V =rewards.copy()  # Initialize value function with immediate rewards
    k = 0
    while True:
        V_new = P @ V + rewards
        if np.linalg.norm(V_new - V) < theta:
            break
        V = V_new
        k += 1
    print("Converged in",k,"iterations.")
    return V


def optimal_policy(rewards,actions,states,neighbors,iterations,tol):
    n_actions, n_states = len(actions), len(states)
    P = np.zeros((n_actions, n_states, n_states))  # transition probabilities
    R = np.zeros((n_actions,n_states))  # immediate reward for taking action a in state s

    for a in actions:
        for s in range(n_states):               
            next_s = neighbors[s][a]
            P[a,s, next_s] = 1.0  # deterministic transition
            if next_s in rewards:
                R[a,s] = rewards[next_s]

    
    V = rewards.copy()
    for it in range(iterations):
        Q = np.zeros((n_actions, n_states))
        for a in range(n_actions):
            Q[a] = R[a] + (P[a] @ V)
        V_new = np.max(Q, axis=0)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    # extract greedy policy
    Q = np.zeros((n_actions, n_states))
    for a in range(n_actions):
        Q[a] = R[a] + (P[a] @ V)
    policy = np.argmax(Q, axis=0)  # deterministic action per state
    return V, policy


def createReward_list(no_states,A_reward,B_reward,r_nt):
    rewards = np.zeros(no_states)
    for s in range(no_states):
    
        if s ==1 or s ==(no_states-2):
            rewards[s] =(A_reward + r_nt)/2
        if s ==4 or s == 3:
            rewards[s] =(B_reward + r_nt)/2
        if s == 5 or s==2:
            rewards[s] = r_nt
    return rewards

def main():
    no_states = 8
    A_reward = 20
    B_reward = 0
    R_nt = 0
    P = np.zeros((no_states,no_states))

    # Absorbing states
    P[0, 0] = 1
    P[7, 7] = 1

    # Transitions for non-terminal states
    neighbors = {
    0: [0, 0],        # absorbing A
    1: [0, 2],
    2: [1, 3],
    3: [2, 7],
    4: [5, 7],
    5: [6, 4],
    6: [0, 5],
    7: [7, 7]         # absorbing B
    }


    for s in range(1,7):
        for nxt in neighbors[s]:
            P[s, nxt] = 0.5
            
            
    rewards = np.array([0,10,0,0,0,0,10,0])
    rewards = createReward_list(no_states,A_reward,B_reward,R_nt)
    #Question 1
    V_pi = MDP_value_iteration(P, rewards)
    print("Optimal Value Function:")    
    print(V_pi)
    
    #Question 2
    V_pi, policy = optimal_policy(rewards, actions=[0,1], states=list(range(8)), neighbors=neighbors, iterations=1000, tol=1e-6)
    print("\n\nOptimal Value Function from Optimal Policy:")
    print(V_pi)
    print("Optimal Policy (0-indexed):")
    print(policy)
    
    #Question 3
    R_nt = -1
    rewards = createReward_list(no_states,A_reward,B_reward,R_nt)
    V_pi, policy = optimal_policy(rewards, actions=[0,1], states=list(range(no_states)), neighbors=neighbors, iterations=1000, tol=1e-6)
    print("\n\nWith negative non-terminal reward -1:") 
    print("Optimal Value Function from Optimal Policy:")
    print(V_pi)
    print("Optimal Policy (0-indexed):")
    print(policy)
    
    #Question 4
    R_nt = -10
    rewards = createReward_list(no_states,A_reward,B_reward,R_nt)
    V_pi, policy = optimal_policy(rewards, actions=[0,1], states=list(range(no_states)), neighbors=neighbors, iterations=1000, tol=1e-6)
    print("\n\nWith more negative non-terminal reward -10:")
    print("Optimal Value Function from Optimal Policy:")
    print(V_pi)
    print("Optimal Policy (0-indexed):")
    print(policy)
if __name__ == "__main__":
    main()    