import numpy as np
import nashpy as nh
import matplotlib.pyplot as plt
import random


def best_responses(matrix,rowplayer,emp_col):    
    if rowplayer:
        vals = matrix.dot(emp_col)  
    else:   
        vals = matrix.T.dot(emp_col)
        # expected payoff for each row action
    return np.where(np.isclose(vals, vals.max()))[0]


def fictitious_play(row_matrix,column_matrix,T ):
    C_row = np.zeros(row_matrix.shape[0])
    C_col = np.zeros(column_matrix.shape[1])
    a_row = random.randint(0,len(C_row)-1)  
    a_col =  random.randint(0,len(C_col)-1)
    C_row[a_row] += 1; C_col[a_col] += 1
    for t in range(1, T):
        emp_row = C_row / C_row.sum()
        emp_col = C_col / C_col.sum()
        br_row = best_responses(row_matrix,True,emp_col)
        br_col = best_responses(column_matrix,False,emp_row)
        a_row = random.choice(br_row)
        a_col = random.choice(br_col)
        C_row[a_row] += 1
        C_col[a_col] += 1
    return C_row / C_row.sum(), C_col / C_col.sum()

def makeFrequencyhistogram(data, num_bins,player):
    actions = ['T','M','B'] if player=="1" else ['L','C','R']

    plt.bar(actions, data)
    plt.title("Player "+player+": Empirical distribution over actions")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.xticks(actions)
    plt.ylim(0, 1)
    plt.show()
    plt.savefig("A2_2_player_"+player+".png")
  

def main():
        
    A = np.array([[1,7,1], [1,4,3],[3,1,4]])
    B = np.array([[6,8,12], [1,2,1],[5,1,1]])

    G = nh.Game(A, B)
    equilibria = G.vertex_enumeration()
    for eq in equilibria:
        print('Equilibrium strategies for row and col player')
        print(eq)
    print('\nFictitious Play Simulations:\n')
   
    emp_row, emp_col = fictitious_play(A, B, T=10000)
    
    print(f'Empirical row strategy: {emp_row}')
    print(f'Empirical column strategy: {emp_col}\n')

    makeFrequencyhistogram(emp_row, 3,"1")    
    makeFrequencyhistogram(emp_col, 3,"2")
        
if __name__ == "__main__":
    main()
