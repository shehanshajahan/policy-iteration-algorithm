# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.

## POLICY ITERATION ALGORITHM
-> Step1 :
We are going to do policy evaluation of each state to get the state value function where the initial policy is defined randomly to the mdp.

-> Step2:
Once we obtain convergence in the policy evaluation then implement policy improvement where we are going to find best optimal policy until the previous and current policy are same.
</br>
</br>

## POLICY IMPROVEMENT FUNCTION
#### Name : Shehan Shajahan
#### Register Number : 212223240154
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to improve the given policy
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
          new_pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi
```
## POLICY ITERATION FUNCTION
#### Name : Shehan Shajahan
#### Register Number : 212223240154
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
   random_actions=np.random.choice(tuple(P[0].keys()),len(P))
   pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
   while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
   return V, pi
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
<img width="467" height="147" alt="image" src="https://github.com/user-attachments/assets/f2054c18-6043-43e1-a911-577fc6d6bdba" />
<img width="619" height="15" alt="image" src="https://github.com/user-attachments/assets/c5e1697d-997d-4ded-8067-b55da25aef1e" />
<img width="475" height="142" alt="image" src="https://github.com/user-attachments/assets/960a1a76-bfae-4e09-ae9d-3511b1f3734c" />




### 2. Policy, Value function and success rate for the Improved Policy
<img width="457" height="150" alt="image" src="https://github.com/user-attachments/assets/8c33ea69-218d-4f45-af0a-3402d068103c" />
<img width="636" height="26" alt="image" src="https://github.com/user-attachments/assets/3a74b5f6-bd21-40a4-b12a-16310cac07c7" />
<img width="492" height="149" alt="image" src="https://github.com/user-attachments/assets/a74ec34a-eb4c-4211-a8dc-b4475934deb1" />




### 3. Policy, Value function and success rate after policy iteration
<img width="768" height="146" alt="image" src="https://github.com/user-attachments/assets/e529aa12-15c0-4b4d-b978-d31db8f5e95e" />
<img width="628" height="30" alt="image" src="https://github.com/user-attachments/assets/3fcd28db-336b-44da-92c0-e36cb7f5ed9f" />
<img width="820" height="122" alt="image" src="https://github.com/user-attachments/assets/89f90593-2479-415f-8492-afeef251b84a" />


## RESULT:
Thus, The Python program to find the optimal policy for the given MDP using the policy iteration algorithm is successfully executed.
