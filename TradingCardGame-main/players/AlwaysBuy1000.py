from player import Player
class AlwaysBuy1000(Player):
    def decide(self, visible_cards, buy_price, sell_price):
        return (True, 1000)

    def reveal(self, cards):
        pass
    

def buildAlwaysBuy1000(card_count, round, budget):
    return AlwaysBuy1000("Always Buy 1000", card_count, round, budget)
