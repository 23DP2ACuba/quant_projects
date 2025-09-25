import numpy as np
import pandas as pd
np.random.seed(42)

states = ["Current", "30-59 days", "60-89 days", "90+ days"]
n_loans = 1000
n_months = 12

initial_distribution = [0.85, 0.08, 0.05, 0.02]
loan_states = np.random.choice(states, size=n_loans, p=initial_distribution)

t_mtx = np.array([
    [0.99, 0.01, 0.00, 0.00],
    [0.25, 0.70, 0.05, 0.00],
    [0.15, 0.00, 0.80, 0.05],
    [0.01, 0.00, 0.00, 0.99]
])

loan_history = []
for i in range(n_months):
  next_states = []
  for loan in loan_states:
    i_current_state = states.index(loan)
    next_state = np.random.choice(states, p=t_mtx[i_current_state])
    next_states.append(next_state)

  loan_states = next_states
  loan_history.append(loan_states.copy())

loan_df = pd.DataFrame(loan_history, columns=[f"loan_{i}" for i in range(n_loans)])
loan_df.index.name = "Month"

transitions = np.zeros((len(states), len(states)))

for t in range(len(loan_history)-1):
  current_states = loan_history[t]
  next_state = loan_history[t+1]
               
  for curr, next_state in zip(current_states, next_states):
    i = states.index(curr)
    j = states.index(next_state)
    transitions[i, j] += 1

mle_mtx = transitions / transitions.sum(axis=1)[:, np.newaxis]

print(loan_df.iloc[:5, :6])
print("Distribution")
print(pd.Series(loan_history[0]).value_counts())

print("MLE")
print(np.round(mle_mtx, 3))
print("Original transitions")
print(t_mtx)
print("Diff")
print(np.round(np.abs(mle_mtx - t_mtx), 3))

