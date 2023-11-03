from typing import Callable
from src.match_types_library.matches_fixed_checkpoint import main as match_same_checkpoint
from src.match_types_library.matches_self_checkpoint import main as match_diff_checkpoints
from src.match_types_library.matches_sizes import main as match_final_checkpoint
from src.match_types_library.matches_solver import main as match_solver_agent
from src.match_types_library.matches_solver_on_solver import main as match_solver_solver



matches_types_lib: dict = {"fixed": src.match_types_library.matches_fixed_checkpoint.main, ...}

def print_menu():
    print("Welcome to the game!")
    print("Please select an option:")
    print("1. Start a new game")
    print("2. Load a saved game")
    print("3. Quit")

def get_user_choice() -> Callable:
    choice = input("Enter your choice (1-3): ")
    while choice not in ['1', '2', '3']:
        choice = input("Invalid choice. Please enter a valid choice (1-3): ")
    return int(choice)

def run():
    # Example usage
    print_menu()
    user_choice = get_user_choice()
    if user_choice == 1:
        start_new_game()
    elif user_choice == 2:
        load_saved_game()
    else:
        quit_game()