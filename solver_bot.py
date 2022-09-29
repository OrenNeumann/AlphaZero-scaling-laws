"""
Create benchmark players using either a solver (for connect Four) or a combination
of a good agent and an opening book (for Pentago).
See code comments for instructions on how to set up the solvers.

This is a bot with the same interface as an MCTSBot, but that instead just calls a solver
to produce perfect play policy vectors.

NOTES:
    * We chose to pick a random move from the optimal ones inside SolverBot, rather than
        pass on all optimal moves and let one of them be picked by _my_play_game.
        The reason is that the best_child() function has no RNG and will always pick
        the first optimal move.
    * The connect four solver requires the opening book (7x6.book) in the directory
        where the code is executed (so /work/ in the cluster for example).
        Otherwise it will run extremely slow (without throwing exceptions!).
"""

import numpy as np
import subprocess
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from pentago_solver import pentago_solver
import AZ_helper_lib as AZh

config = AZh.Config(
    game="pentago",
    MC_matches=False,
    path=None,
    path_model_1=None,
    path_model_2=None,
    checkpoint_number_1=None,
    checkpoint_number_2=None,
    use_solver=False,
    logfile=None,
    learning_rate=0,
    weight_decay=0,
    temperature=None,
    evaluators=None,
    uct_c=2,
    max_simulations=3000,
    policy_alpha=0.5,
    evaluation_games=None,
    evaluation_window=None,

    nn_model=None,
    nn_width=None,
    nn_depth=None,
    observation_shape=None,
    output_size=None,

    quiet=True,
)


def connect_four_solver(state, full_info=False):
    """ Produces an optimal policy for the current state.
        Returns: An array of optimal moves, i.e. moves that received the
            maximal score from the solver.
        if 'full_info' is set to True, returns a policy vector and value estimation.
            The policy is a uniform distribution over optimal moves.

        This code uses the open source solver available in:
        https://connect4.gamesolver.org/

        In order to use this, download the Github repo:
        https://github.com/PascalPons/connect4
        Hit 'make' and copy the call_solver.sh script into the new folder.
        Then download the openings book here: (7x6.book)
        https://github.com/PascalPons/connect4/releases/tag/book
        and place it in the parent directory.
        """
    moves = ""
    for action in state.full_history():
        # The solver starts counting moves from 1:
        moves += str(action.action + 1)

    # Call the solver and get an array of scores for all moves.
    # Optimal legal moves will have the highest score.
    out = subprocess.run(["./connect4-master/call_solver.sh", moves], capture_output=True)
    out = out.stdout.split()
    if moves == "":
        scores = np.array(out, dtype=int)
    else:  # ignore 1st output (=moves):
        scores = np.array(out[1:], dtype=int)
    if full_info:
        mask = state.legal_actions_mask()
        p = np.extract(mask, scores)
        if (p > 0).any():  # Win
            v = 1
            p[p < np.amax(p)] = 0
            p[p != 0] = 1
        elif (p == 0).any():  # Draw
            v = 0
            p = (p == 0).astype(int)
        else:  # Loss
            v = -1
            p[p < np.amax(p)] = 0
            p[p != 0] = 1

        p = p / sum(p)
        return v, p
    else:
        return np.argwhere(scores == np.amax(scores)).flatten()  # All optimal moves


class SolverBot:
    """A perfect-play bot that uses a game solver.
        For pentago it uses a combination of a good agent with 10 times
        the normal MCTS steps, combined with a solver up to move 17."""

    def __init__(self, game):
        self.use_base_bot = False
        if str(game) == 'connect_four()':
            self.solver = connect_four_solver
        elif str(game) == 'pentago()':
            self.solver = pentago_solver
            self.use_base_bot = True
            path = './models/pentago/q_5_3/'  # The best agent trained.
            model = AZh.load_model_from_checkpoint(config, path=path)
            az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
            self.base_bot = mcts.MCTSBot(
                game,
                2,
                3000,
                az_evaluator,
                child_selection_fn=mcts.SearchNode.puct_value)
        else:
            raise ValueError("No solver supported for game " + str(game))

    def mcts_search(self, state):
        """ Returns a search node with children.
            The node will have no meaningful parameters other than its list
            of children, and the children will only contain an explore_count.
            All children except one have explore_count set to 0.
            One child, chosen randomly from all optimal moves, has explore_count=1.
            This ensures that move will be picked by best_child(), and that the policy
            vector generated will be a one-hot encoding of the chosen child.
            """
        if self.use_base_bot:
            base_root = self.base_bot.mcts_search(state)
            if state.move_number() > 17:
                return base_root
            optimal_moves = self.solver(state)
            # For every child of the root node, set their explore
            # count to 0 if the move is not optimal.
            # If it is optimal, make sure it has a least an explore
            # count of 1 to avoid having no explored children.
            for node in base_root.children:
                if node.action in optimal_moves:
                    if node.explore_count == 0:
                        node.explore_count = 1
                else:  # The node is a suboptimal move
                    node.explore_count = 0
            return base_root

        chosen_move = np.random.choice(self.solver(state))

        num_actions = state.get_game().num_distinct_actions()
        root = mcts.SearchNode(None, None, None)
        root.explore_count = 1

        for action in range(num_actions):
            child = mcts.SearchNode(action, None, None)
            if action == chosen_move:
                child.explore_count = 1
            root.children.append(child)

        return root
