from abc import ABC, abstractmethod

class Player(ABC):
    def __init__(self, name, card_count, rounds, budget):
        self.card_count = card_count
        self.rounds = rounds
        self.name = name
        self.budget = budget

    def __repr__(self):
        return self.name

    def set_budget(self, budget):
        """This will get automatically called, do not touch it"""
        self.budget = budget

    @abstractmethod
    def decide(self, visible_cards, buy_price, sell_price):
        """return a tuple of (buy/sell, count), where 0 = sell and 1 = buy"""

    @abstractmethod
    def reveal(self, cards):
        """shows the player what cards there are"""
        