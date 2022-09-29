# AlphaZero-Scaling
Code and data for 'Scaling Laws for a Multi-Agent Reinforcement Learning Model'.

## Prerequisites
The code should be run with Python (version 3.6.8) using open-spiel (version 1.0.0) and tensorflow (version 2.6.0).

In order to run Connect Four matches against a solver, you must install the open source [solver package](https://github.com/PascalPons/connect4/tree/book) seperately. See `solver_bot.py` documentation for detailed instructions.

In order to calculate Elo scores, you must install the open source [BayesElo Python API](https://github.com/yytdfc/Bayesian-Elo). See `get_bayeselo.py` documentation for detailed instructions.

## Code overview
`train_models.py` is the starting point. This code can be used to train AlphaZero agents with the same hyperparameters used in the paper.

There are two main codes needed to generate the match data required in order to calculate the Elo scores of all agents:
- `matches_fixed_checkpoint.py` is used to run matches on a training slice, running games between all pairs of agents trained for a specified number of training steps.
- `matches_self_checkpoint.py` is used to run matches on an agent slice, running games between all pairs of checkpoints of a single agent.

The Elo scores presented in the paper can be replicated by running matches on all training slices and all agent slices, with 800 games for every player pair, then running `get_elo_scores.py` to generate the Elo rating. 

To generate Elo scores relative to the game solver benchmarks, you should use `matches_solver.py`. Instructions on how to set up the solvers is provided in `solver_bot.py` and `pentago_solver.py`, where the solver players are implemented. The resulting matches vector should be fed to BayesElo.

FLOPs counts for all models were generated with `get_flops.py`.

The main part of the code, where we implemented all functionalities missing in OpenSpiel, is found in `AZ_helper_lib.py`.

## Data
All models trained on Connect Four and Pentago are available in an individual release.
Each agent folder contains all checkpoints used to generate the plots in the paper.
Extract the models in a directory named `matches` for compatibility with the code, or alternatively specify the desired directory in the code itself.
