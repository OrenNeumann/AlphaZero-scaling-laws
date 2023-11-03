from absl import app
from open_spiel.python.utils import spawn


import src.cli as cli

if __name__ == '__main__':
    user_input = cli.run()
    with spawn.main_handler():
        app.run(user_input)

        
    
