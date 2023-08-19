def connect_four_solver(state, full_info=False, temperature=None):
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
    if full_info and (temperature is not None):
        raise Exception('full_info not supported with temperature.')
    moves = ""
    for action in state.full_history():
        # The solver starts counting moves from 1:
        moves += str(action.action + 1)

    # Call the solver and get an array of scores for all moves.
    # Optimal legal moves will have the highest score.
    out = subprocess.run(["./connect4-master/call_solver.sh", moves],  stdout=PIPE, stderr=PIPE)
    out = out.stdout.split()
    if moves == "":
        scores = np.array(out, dtype=int)
    else:  # ignore 1st output (=moves):
        scores = np.array(out[1:], dtype=int)

    if full_info:  # Return policy and value.
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
    elif (temperature is not None) and (temperature != 0):  # Return policy with temperature noise.
        if temperature < 0:
            raise ValueError('Temperature must be non-negative.')
        # Set illegal move scores to -inf:
        scores = np.array(scores, dtype=float)
        scores[scores == -1000] = -np.inf
        # Return the softmax of scores divided by temperature:
        exponents = scores/temperature
        if temperature == np.inf:  # Avoid inf/inf
            exponents[exponents==None] = 0
        return softmax(exponents)
    else:  # Return one-hot vector of the best move (default).
        return np.argwhere(scores == np.amax(scores)).flatten()  # All optimal moves

