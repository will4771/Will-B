from game import Game
import random
from players.AlwaysBuy1 import buildAlwaysBuy1
from players.AlwaysSell1 import buildAlwaysSell1
from players.AlwaysBuy1000 import buildAlwaysBuy1000
from players.algo1 import buildalgo1
from players.algo2 import buildalgo2
from players.algo3 import buildalgo3
from players.algo4 import buildalgo4
from players.algo5 import buildalgo5


ALL_FACTORIES = [
    buildalgo1,
    
    buildalgo3,
    buildalgo4,
    buildalgo5
]


def one_game():
    game = Game(card_count=4, rounds=500)
    game.setup(ALL_FACTORIES)
    return game.run()


def main(games_count=1):
    win_count = {}
    for i in range(games_count):
        winner = one_game()

        if winner.name not in win_count:
            win_count[winner.name] = 1
        else:
            win_count[winner.name] += 1
            
        print(f"Game {i} finished: winner = {winner}")
        
    print(f"Complete.")
    display_result(games_count, win_count)


def display_result(total_count, win_count):
    print("=======================================")
    print("Win count:")
    players = list(sorted(win_count.keys(), key=lambda x: win_count[x], reverse=True))
    for p in players:
        print(f"{p}: {win_count[p]}, ({round(100 * win_count[p] / total_count, 2)}%)")


if __name__ == "__main__":
    main(games_count=10)
