from player import Player
class Will_2(Player):
    def __init__(self, name, card_count, rounds, budget):
        self.card_count = card_count
        self.rounds = rounds
        self.name = name
        self.budget = budget

    def __repr__(self):
        return self.name

    def decide(self, visible_cards, buy_price, sell_price):
        """return a tuple of (buy/sell, count), where 0 = sell and 1 = buy"""

        N = 52
        X = 416
        pred = 0
        self.card_count = 3

        buy_sell = True  ## buy 
        units = 0 

        number_cards_visable = len(visible_cards)

        N -= number_cards_visable
        
        for i in range(number_cards_visable):
            X -= visible_cards[i]
            pred += visible_cards[i]

        E = X/N

        for j in range(self.card_count - number_cards_visable):
            pred += E

            X -= E
            N -= 1

            E = X/N

        if buy_price <= pred:
             
            buy_sell = True
            diff = sell_price - pred
            per = ((diff / 10)**2)**1/2
            if per > 1:
                per = 1
            units = (per *((self.budget) / buy_price) ) - 2

            return(buy_sell,units)
        
        if sell_price >= pred:

            buy_sell = False
            diff = sell_price - pred
            per = ((diff / 8.5)**2)**1/2

            if per > 1:
                per = 1
            units = (per * ((self.budget) / sell_price) )- 2

            return(buy_sell,units)
        
        if sell_price < pred < buy_price:
            return (buy_sell, 0)

    def reveal(self, cards):
        """shows the player what cards there are"""
        

def buildWill_2(card_count, round, budget):
    return Will_2("Will 2", card_count, round, budget)
    