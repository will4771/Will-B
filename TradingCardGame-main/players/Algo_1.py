from player import Player
from itertools import combinations
import statistics
import math


class Andy(Player):
    def __init__(self, name, card_count, rounds, budget, tolerance=0.01, alpha=0.2, estimate_expected=True):
        super().__init__(name, card_count, rounds, budget)
        self.cards = []
        self.tolerance = tolerance
        self.reset_cards()

        self.e_cache = {}

        self.estimate_expected = estimate_expected
        self.alpha = alpha
    
    def reset_cards(self):
        self.cards.clear()
        for i in range(2, 15):
            self.cards += [i] * 4

    def max_number(self, price):
        # add tolerance
        return self.budget / price - self.tolerance

    def expected_value(self, n):
        if self.estimate_expected:
            return statistics.mean(self.cards) * (self.card_count - n)
        else:
            return statistics.mean(map(sum, combinations(self.cards, self.card_count - n)))
        
    def decide(self, visible_cards, buy_price, sell_price):
        n = len(visible_cards)
        if n == self.card_count:
            e = sum(visible_cards)
            if e > buy_price:
                return (True, self.max_number(buy_price) - 1)
            elif e < sell_price:
                return (False, self.max_number(sell_price) - 1)
            else:
                return (True, 0)

        if tuple(visible_cards) in self.e_cache:
            e = self.e_cache[tuple(visible_cards)]
        else:
            self.reset_cards()
            # print("---------------------------------------------------")
            # print(visible_cards)
            # print(f"Buy: {buy_price}")
            # print(f"Sell: {sell_price}")

            for i in visible_cards:
                self.cards.remove(i)
            e = self.expected_value(n) + sum(visible_cards)
        
        if sell_price < e < buy_price:
            return (True, 0)        
        
        if e > buy_price:
            diff = e - buy_price
        else:
            diff = e - sell_price     

        a = math.tanh(self.alpha * diff)
        number = a * self.max_number(buy_price if a > 0 else sell_price)
        # print(f"Expected: {e}")
        # print("---------------------------------------------------")
        return number > 0, abs(number)

    def reveal(self, cards):
        pass


def buildAndy(card_count, round, budget):
    # calculate optimal value of alpha
    alpha = min(0.7, ((round - 50) / 200 * 0.4 + 0.3))
    return Andy("Andy", card_count, round, budget, tolerance=0.01, alpha=alpha, estimate_expected=card_count>3)

