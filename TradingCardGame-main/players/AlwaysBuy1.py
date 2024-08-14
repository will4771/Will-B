from player import Player
class AlwaysBuy1(Player):
    def decide(self, visible_cards, buy_price, sell_price):
        return (True, 1)

    def reveal(self, cards):
        pass
    

def buildAlwaysBuy1(card_count, round, budget):
    return AlwaysBuy1("Always Buy 1", card_count, round, budget)
