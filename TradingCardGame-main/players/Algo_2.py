from player import Player
import random
 

class Arthur(Player):
    def __init__(self, name, card_count, rounds, budget, base_buy_threshold_high, base_sell_threshold_low):
        super().__init__(name, card_count, rounds, budget)
        self.remaining_cards = [i for i in range(2, 15) for _ in range(4)]
        self.base_buy_threshold_high = base_buy_threshold_high
        self.base_sell_threshold_low = base_sell_threshold_low
        self.current_buy_threshold_high = base_buy_threshold_high
        self.current_sell_threshold_low = base_sell_threshold_low
 
    def adjust_thresholds(self):
        # Adjust buy and sell thresholds dynamically based on remaining budget
        budget_ratio = self.budget / 500  # Assuming starting budget is 500
        scaling_factor = 0  # Smaller factor for more conservative adjustments
        self.current_buy_threshold_high = self.base_buy_threshold_high * (1 + budget_ratio * scaling_factor)
        self.current_sell_threshold_low = self.base_sell_threshold_low * (1 - budget_ratio * scaling_factor)
 
    def decide(self, visible_cards, buy_price, sell_price):
        self.adjust_thresholds()  # Adjust thresholds before making a decision
 
        visible_sum = sum(visible_cards)
        for card in visible_cards:
            if card in self.remaining_cards:
                self.remaining_cards.remove(card)
 
        unseen_card_count = self.card_count - len(visible_cards)
        expected_value_per_card = sum(self.remaining_cards) / len(self.remaining_cards) if self.remaining_cards else 0
        expected_unseen_sum = unseen_card_count * expected_value_per_card
        expected_sum_cards = visible_sum + expected_unseen_sum
 
        # Slight randomness applied to risk management (optional)
        risk_tolerance = random.uniform(1, 1)
 
        if expected_sum_cards > self.current_buy_threshold_high * buy_price * risk_tolerance:
            max_affordable_units = self.budget // buy_price
            units_to_buy = max_affordable_units  # Buy as many as possible
            return (True, units_to_buy)
 
        elif expected_sum_cards < self.current_sell_threshold_low * sell_price * risk_tolerance:
            units_to_sell = self.budget // sell_price  # Sell as many as possible
            return (False, units_to_sell)
 
        # If within the margins, make smaller trades based on proportional expected profit
        if expected_sum_cards > buy_price:
            budget_fraction = (expected_sum_cards - buy_price) / (self.current_buy_threshold_high * buy_price - buy_price)
            units_to_buy = int(self.budget * budget_fraction // buy_price)
            return (True, max(units_to_buy, 1))
 
        elif expected_sum_cards < sell_price:
            units_to_sell = int((sell_price - expected_sum_cards) / (sell_price - self.current_sell_threshold_low * sell_price))
            return (False, max(units_to_sell, 1))
 
        return (True, 1)  # Skip the turn if the opportunity isn't clear
 
    def reveal(self, cards):
        for card in cards:
            if card in self.remaining_cards:
                self.remaining_cards.remove(card)
 

def buildArthur(card_count, rounds, budget):
    return Arthur("Arthur", card_count, rounds, budget, 1.51, 0.87)