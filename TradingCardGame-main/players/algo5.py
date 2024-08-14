from player import Player
class algo5(Player):
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
            ## exponetial increase of per, could do with optimising for a ... 
            a = 0.5
            buy_sell = True
            diff = sell_price - pred
            per = ((diff / 10)**2)**0.5

            adj = per**a

            if adj > 1:
                adj = 1
            units = (adj *((self.budget) / buy_price)) - 1/2

            return(buy_sell,units)
        
        if sell_price >= pred:
            ## exponetial increase of per
            a = 0.5
            buy_sell = False
            diff = sell_price - pred
            per = ((diff / 8.50)**2)**0.5

            adj = per**a

            if adj > 1:
                adj = 1

            units = (adj * ((self.budget) / sell_price) ) - 1/2

            return(buy_sell,units)
        
        if sell_price < pred < buy_price:
            return (buy_sell, 0)

    def reveal(self, cards):
        """shows the player what cards there are"""
    

def buildalgo5(card_count, round, budget):
    return algo5("algo5", card_count, round, budget)

    