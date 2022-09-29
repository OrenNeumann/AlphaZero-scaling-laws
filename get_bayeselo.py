#!/usr/bin/env python
# encoding: utf-8

from Bayesian_Elo import bayeselo
import numpy as np

"""
Code for interfacing with the BayesElo Python API found here:
https://github.com/yytdfc/Bayesian-Elo
Download the repository, place it under the parent folder and hit make.


How this code works:
Feed in the games to 'r' in the form of:
r.append(x, y, z)
where:
x = number of first player
y = number of second player
z = game outcome: 0 loss, 1 tie, 2 win (for player x)

Then add player names to the EloRating function.
"""


def get_bayeselo(matches):
    """ Returns the Elo ratings of all agents with BayesElo."""
    dims = matches.shape
    if matches.ndim != 4:
        raise Exception('Only supports 4D tensors.')
    else:
        n_agents = matches.shape[0] * matches.shape[2]
    agents = list(map(str, list(range(n_agents))))
    r = bayeselo.ResultSet()

    print('Loading games...')
    for i in range(dims[0]):
        for j in range(dims[2]):
            player_1 = i * dims[2] + j
            for m in range(dims[0]):
                for n in range(dims[2]):
                    player_2 = m * dims[2] + n
                    p = matches[i, m, j, n]
                    if p is np.ma.masked:
                        continue
                    p = int(p * 400)
                    for k in range(400):
                        # Interpret match results as 800 games that resulted in a win/loss:
                        r.append(player_1, player_2, int(k < p) * 2)
                        r.append(player_2, player_1, int(k > p) * 2)

    print('Calculating rating...')
    e = bayeselo.EloRating(r, agents)
    e.offset(1000)
    e.mm()
    e.exact_dist()
    print(e)
    x = e.__str__().split("\n")
    table = []
    for row in x:
        table.append(row.split())
    table = np.array(table[:-1])
    agent_order = table[:, 1]
    elo = table[:, 2]
    agent_order = agent_order[1:].astype(int)
    elo = elo[1:].astype(float)
    elo = elo[agent_order.argsort()]
    return elo
