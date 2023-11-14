from typing import Callable
from src.match_types_library.matches_fixed_checkpoint import main as match_same_checkpoint
from src.match_types_library.matches_self_checkpoint import main as match_diff_checkpoints
from src.match_types_library.matches_sizes import main as match_final_checkpoint
from src.match_types_library.matches_solver import main as match_solver_agent
from src.match_types_library.matches_solver_on_solver import main as match_solver_solver

matches_types_lib: dict = {1: match_final_checkpoint,
                           2: match_same_checkpoint, #CP
                           3: match_diff_checkpoints, # i, j
                           4: match_solver_agent,
                           5: match_solver_solver}


def print_menu():
    print("Select a game:")
    print("1. Connect Four")
    print("2. Pentago")
    print("3. Oware")
    game_choice = get_user_choice(3)

    print("Select the type of matches you wish to run:")
    print("1. Match fully-trained agents")
    print("2. Match partially-trained agents")
    print("3. Match different checkpoints of the same agent")
    print("4. Match agents with a solver")
    print("5. Match solvers with solvers")
    match_choice = get_user_choice(5)

    return game_choice, match_choice


def get_user_choice(n) -> int:
    choice = input("\nEnter your choice (1-{}): ".format(n))
    while choice not in [str(i + 1) for i in range(n)]:
        choice = input("Invalid choice. Please enter a valid choice (1-{}): ".format(n))
    return int(choice)


def run():
    # Example usage
    print_menu()
    if user_choice == 1:
        start_new_game()
    elif user_choice == 2:
        load_saved_game()
    else:
        quit_game()
