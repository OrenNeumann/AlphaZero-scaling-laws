"""
Calculate the Elo scores of agents from match results.

This is done by loading all match data to a 4-D tensor:
matches[first_agent, second_agent, first_checkpoint, second_checkpoint]
and using it for Elo rating calculation with BayesElo.

The BayesElo API should be installed from source, see 'get_bayeselo.py' for documentation.

Before running this code, run matches for all agents and checkpoints with
'matches_fixed_checkpoint.py' and 'matches_self_checkpoint'.
"""
import numpy as np
from get_bayeselo import get_bayeselo

# Specify the game:
game = 'connect_four'

if game == 'connect_four':
    use_f_models = True
elif game == 'pentago':
    use_f_models = False
else:
    raise NameError('Invalid game.')

n_copies = 6
max_q_self = 6
n_copies_self = 6
checkpoints = [20, 30, 50, 70, 100, 150, 230, 340, 510, 770, 1150, 1730, 2590, 3880, 5820, 8730, 10000]
checkpoints = np.asarray(checkpoints)
# Create a list of all trained models
nets = []
max_q = 6
min_f = 0
max_f = 5
# Enumerate all models. The order is: First by size, then by copy number.
for i in range(max_q + 1):
    for j in range(n_copies):
        nets.append('q_' + str(i) + '_' + str(j))

    if (min_f <= i <= max_f) and use_f_models:
        for j in range(n_copies):
            nets.append('f_' + str(i) + '_' + str(j))

# create key for model numeration:
model_num = dict(zip(nets, range(len(nets))))
n_checks = len(checkpoints)
n_models = len(model_num)
matches = np.zeros([n_models, n_models, n_checks, n_checks])
matches[:] = 1000

models_self = []
# Enumerate self-matches models
for i in range(max_q_self + 1):
    for j in range(n_copies_self):
        models_self.append('q_' + str(i) + '_' + str(j))

n_checkpoints = len(checkpoints)

for i in range(len(models_self)):
    model = models_self[i]
    print(model)
    a = np.load('./self_matches/' + game + '/model_' + str(model) + '.npy')
    a += np.diag([1000] * len(a))
    matches[i, i, :, :] = a

for i in range(len(checkpoints)):
    cp = checkpoints[i]
    print(cp)
    a = np.load('./fixed_matches/' + game + '/checkpoint_' + str(cp) + '.npy')
    a = a[:n_models, :n_models]
    a += np.diag([1000] * len(a))
    matches[:, :, i, i] = a

# Mask all missing matches (set to 1000).
mask = matches > 1
matches = np.ma.masked_array(matches, mask)

# Use BayesElo to turn the matches tensor into a matrix of Elo scores.
# Rows stand for agent number, columns stand for checkpoint number.
elos = get_bayeselo(matches)
bayes_e = np.zeros([n_models, n_checks])
for i in range(n_models):
    for j in range(n_checks):
        ind = i * n_checks + j
        bayes_e[i, j] = elos[ind]
# 'bayes_e' contains all Elo scores, normalized so that the lowest agents has a score of 0.
bayes_e = bayes_e - bayes_e.min()
print('Elo scores:')
print(bayes_e)
