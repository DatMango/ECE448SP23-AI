'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition_matrix(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    #raise RuntimeError("You need to write this part!")
    P = np.zeros([model.M, model.N, 4, model.M, model.N])
    for r in range(0, model.M):
        for c in range(0, model.N):
            for a in range(0, 4):
                if model.T[r, c]:
                    P[r, c, a, :, :] = 0
                    continue
                if a == 0:
                    if c-1 < 0 or model.W[r, c-1]:
                        P[r, c, 0, r, c] += model.D[r, c, 0]
                    else:
                        P[r, c, 0, r, c-1] += model.D[r, c, 0]

                    if r+1 >= model.M or model.W[r+1, c]:
                        P[r, c, 0, r, c] += model.D[r, c, 1]
                    else:
                        P[r, c, 0, r+1, c] += model.D[r , c, 1]

                    if r-1 < 0 or model.W[r-1, c]:
                        P[r, c, 0, r, c] += model.D[r, c, 2]
                    else:
                        P[r, c, 0, r-1, c] += model.D[r, c, 2]
                elif a == 1:
                    if r-1 < 0 or model.W[r-1, c]:
                        P[r, c, 1, r, c] += model.D[r, c, 0]
                    else:
                        P[r, c, 1, r-1, c] += model.D[r, c, 0]
                    
                    if c-1 < 0 or model.W[r, c-1]:
                        P[r, c, 1, r, c] += model.D[r, c, 1]
                    else:
                        P[r, c, 1, r, c-1] += model.D[r, c, 1]
                    
                    if c+1 >= model.N or model.W[r, c+1]:
                        P[r, c, 1, r, c] += model.D[r, c, 2]
                    else:
                        P[r, c, 1, r, c+1] += model.D[r, c, 2]
                elif a == 2:
                    if c+1 >= model.N or model.W[r, c+1]:
                        P[r, c, 2, r, c] += model.D[r, c, 0]
                    else:
                        P[r, c, 2, r, c+1] += model.D[r, c, 0]
                    
                    if r-1 < 0 or model.W[r-1, c]:
                        P[r, c, 2, r, c] += model.D[r, c, 1]
                    else:
                        P[r, c, 2, r-1, c] += model.D[r, c, 1]
                    
                    if r+1 >= model.M or model.W[r+1, c]:
                        P[r, c, 2, r, c] += model.D[r, c, 2]
                    else:
                        P[r, c, 2, r+1, c] += model.D[r, c, 2]
                else:
                    if r+1 >= model.M or model.W[r+1, c]:
                        P[r, c, 3, r, c] += model.D[r, c, 0]
                    else:
                        P[r, c, 3, r+1, c] += model.D[r, c, 0]
                    
                    if c+1 >= model.N or model.W[r, c+1]:
                        P[r, c, 3, r, c] += model.D[r, c, 1]
                    else:
                        P[r, c, 3, r, c+1] += model.D[r, c, 1]
                    
                    if c-1 < 0 or model.W[r, c-1]:
                        P[r, c, 3, r, c] += model.D[r, c, 2]
                    else:
                        P[r, c, 3, r, c-1] += model.D[r, c, 2]
    return P


def update_utility(model, P, U_current):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    #raise RuntimeError("You need to write this part!")
    U_next = np.zeros_like(U_current)
    for r in range(0, model.M):
        for c in range(0, model.N):
            curr_larg = -float("inf")
            for a in range(0, 4):
                temp = 0
                for r_prime in range(0, model.M):
                    for c_prime in range(0, model.N):
                        temp += P[r, c, a, r_prime, c_prime] * U_current[r_prime, c_prime]
                if temp > curr_larg:
                        curr_larg = temp
            U_next[r, c] = model.R[r, c] + model.gamma * curr_larg

    return U_next


def value_iteration(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    #raise RuntimeError("You need to write this part!")
    P = compute_transition_matrix(model)
    U_current = np.zeros([model.M, model.N])
    U_next = update_utility(model, P, U_current)
    while True:
        if np.all(U_next - U_current < epsilon):
            return U_next
        U_current = U_next
        U_next = update_utility(model, P, U_current)

if __name__ == "__main__":
    import utils
    model = utils.load_MDP('models/small.json')
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
