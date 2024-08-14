from player import Player
class AlwaysSell1(Player):
    def decide(self, visible_cards, buy_price, sell_price):
        return (False, 1)

    def reveal(self, cards):
        pass
    

def buildAlwaysSell1(card_count, round, budget):
    return AlwaysSell1("Always Sell 1", card_count, round, budget)
