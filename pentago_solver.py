"""
Code for interfacing with an open source Pentago solver, available in:
https://perfect-pentago.net/

IMPORTANT NOTE: before you use this code, send an email to Geoffrey Irving
and ask for permission to use his server.
This code will bombard it with queries for game states.
After you have done this, you can remove the safety clause in 'get_backend_dict'.
"""

import numpy as np
import requests
import ast
import copy


n_connection_failure = 0
dict_conversion_failure = 0


def get_backend_dict(num):
    """ Returns a dictionary of children outcomes produced by the website.
        added try/except clauses for internet connection failure and
        failure to parse the HTML page to a dict object."""

    raise Exception('Ask for permission to use the Pentago solver. '
                    'See: \nhttps://perfect-pentago.net/')

    link = 'https://us-central1-naml-148801.cloudfunctions.net/pentago/' + str(num)
    try:
        response = requests.get(link)
        content = response.content
    except OSError as e:
        print(e)
        global n_connection_failure
        n_connection_failure += 1
        print('Connection failure number: ', n_connection_failure)
        return 'failure'

    try:
        return ast.literal_eval(content.decode('utf-8'))
    except SyntaxError as e:
        print(e)
        global dict_conversion_failure
        dict_conversion_failure += 1
        return 'failure'


def board_to_int(board):
    """ Receives a Pentago board and returns a number representation of it.
        Inverse function of int_to_board.
        Follows the instructions given in:
        https://us-central1-naml-148801.cloudfunctions.net/pentago
        Note that ternary/binary numbers should be built from right to left.
        Therefore quadrant and string orders are reversed.

        Input: 6x6 array of the numbers {0,1,2} representing empty/
            black/white respectively."""
    # FIRST STEP: convert each quadrant of the board to a ternary number representation.
    # Rotate for x-major orientation:
    board = np.rot90(board, k=3)
    # Order quadrants y-major (== x-major order pre-rotation):
    quads = [board[:3, :3], board[:3, 3:], board[3:, :3], board[3:, 3:]]
    # Reverse quadrant order:
    quads = quads[::-1]
    ternaries = [''] * 4
    i = 0
    for q in quads:
        for row in q:
            for n in row:
                ternaries[i] += str(int(n))
        # Reverse string order:
        ternaries[i] = ternaries[i][::-1]
        i += 1

    # SECOND STEP: Translate each quadrant's ternary representation to binary
    binaries = []
    for quad in ternaries:
        num = int(quad, 3)
        binaries.append(np.binary_repr(num, width=16))

    # THIRD STEP: Join the binary representations and return an int representing
    # the board position.
    full_num = ''.join(binaries)
    return int(full_num, 2)


def int_to_board(num):
    """ Take a number representation of a Pentago board and return an array
        of the board position.
        Inverse function of board_to_int, look there for documentation."""
    # Int to binary:
    binary_num = np.binary_repr(num, width=64)
    # Each 16 digits of binary to ternary:
    quads = []
    for i in range(4):
        n = int(binary_num[i * 16:(i + 1) * 16], 2)
        n = np.base_repr(n, base=3)
        n = '0' * (9 - len(n)) + n
        quads.append(n)
    # Ternary to array:
    board = np.zeros([6, 6])
    for i in range(4):
        # loop over quadrants
        quad = quads[i]
        x0 = 3 * int(i / 2)
        y0 = 3 * (i % 2)
        count = 0
        for row in range(3):
            for col in range(3):
                board[x0 + row, y0 + col] = quad[count]
                count += 1
    return np.rot90(board, k=3)


def state_to_array(state):
    """ Translate pyspiel state to numpy array.
        '.' ---> 0
        '0' ---> 1
        '@' ---> 2
        """
    lines = str(state).splitlines()
    board = np.zeros([6, 6])
    for row in range(6):
        chars = lines[2 + row][4:-2].split(' ')
        for col in range(6):
            if chars[col] == 'O':
                board[row, col] = 1
            elif chars[col] == '@':
                board[row, col] = 2
    return board


def pentago_solver(state):
    """ Produces an optimal policy for the current state.
        Returns: An array of optimal moves, i.e. moves that received the
            maximal score from the solver.

        This solver utilizes data pulled from this Pentago solver:
        https://github.com/girving/pentago
        The OpenSpiel game state is converted to the number representation
        used by the solver, and information about the score of all current
        moves is pulled from either the openings book (up to move 17) or
        the midgame engine (currently not available).
        The only information available is the minimax score of a game position,
        therefore this function returns a uniform distribution on all moves
        with optimal score.

        Added some safeguards against bugs:
        - sometimes the website output doesn't contain information on all the child nodes.
            if that happens, abort if move_number==17, otherwise query the website again
            for the specific child node.
        - If the website fails to connect or the output was not a dict, abort.
        When aborting, the function returns a positive mask on all actions (all moves are good).
        """

    def state_to_num(s):
        return board_to_int(state_to_array(s))

    board_num = state_to_num(state)
    actions = np.array(state.legal_actions())
    # Get a dictionary of scores of all child nodes from the Pentago solver.
    if state.move_number() < 18:
        # From the backend server:
        moves_dict = get_backend_dict
        # If failed to connect, don't use the solver this turn:
        if moves_dict == 'failure':
            return actions.tolist()
        children = moves_dict(board_num)
    else:
        # From the midgame engine:
        raise Exception('Midgame engine interface not implemented yet.')

    scores = []

    for action in actions:
        child = copy.deepcopy(state)
        child.apply_action(action)
        child_num = state_to_num(child)
        key = str(child_num)
        if key in children:
            scores.append(-children[key])
        elif state.move_number() < 17:
            # State not in the dictionary, query the website again for this state:
            child_dict = moves_dict(child_num)
            if child_dict == 'failure':
                # Bad output, abort
                print('BadWebsiteOutput')
                return actions.tolist()
            scores.append(-child_dict[key])
        else:
            # Can't query move 18, abort
            print('KeyError')
            return actions.tolist()
    # Return all optimal moves:
    return actions[scores == np.amax(scores)]
