import random
from players.AlwaysBuy1 import buildAlwaysBuy1
from players.AlwaysSell1 import buildAlwaysSell1
from players.AlwaysBuy1000 import buildAlwaysBuy1000
from players.Andy import buildAndy
from players.will_1 import buildWill_1
from players.will_2 import buildWill_2
from players.will_3 import buildWill_3
from players.Arthur import buildArthur


class InsufficientFundsException(Exception):
    pass


class Game:
    def __init__(self, 
            card_count=3, 
            rounds=3,
            visible_p=0.25,
            bid_ask_spread=3,
            price_interval=10,
            starting_budget=500,
            reshuffle=True,
            display=False
        ):
        self.cards = []
        self.players = []

        self.money = {}
        self.card_count = card_count
        self.rounds = rounds

        self.visible_p = visible_p
        self.bid_ask_spread = bid_ask_spread
        self.price_interval = price_interval
        self.starting_budget = starting_budget

        self.reshuffle = reshuffle

        self.display = display

    def reset_cards(self):
        self.cards.clear()
        for i in range(2, 15):
            self.cards += [i] * 4
        random.shuffle(self.cards)

    def setup(self, factories):
        self.reset_cards()
        for f in factories:
            self.players.append(f(self.card_count, self.rounds, self.starting_budget))

        for p in self.players:
            self.money[p] = self.starting_budget

    def randomly_pick_cards(self):
        result = []
        for _ in range(self.card_count):
            item = self.cards[0]
            self.cards.pop(0)
            result.append((item, random.random() < self.visible_p))
        return result

    def calculate_profits(self, calculated_price, actual_price, number, buy):
        diff = (calculated_price - actual_price) * number
        if buy:
            return diff
        else:
            return -diff

    def penalty(self, p):
        self.money[p] = 0
        p.set_budget(0)

    def run_one_round(self):
        cards = self.randomly_pick_cards()
        visible_cards = list(map(lambda x: x[0], filter(lambda x: x[1], cards)))
        nums_in_play = list(map(lambda x: x[0], cards))

        target_price = sum(nums_in_play)
        sell_price = random.randint(max(0, target_price - self.price_interval - self.bid_ask_spread // 2),
                                    target_price + self.price_interval - self.bid_ask_spread // 2)
        buy_price  = sell_price + random.randint(1, self.bid_ask_spread)

        for p in self.players:
            try:
                buy, number = p.decide(visible_cards, buy_price, sell_price)
                number = max(0, number)

                actual_price = buy_price if buy else sell_price
                
                if actual_price * number > self.money[p]:
                    raise InsufficientFundsException
                
                profit = self.calculate_profits(target_price, actual_price, number, buy)
                self.money[p] += profit
                p.set_budget(self.money[p])
                p.reveal(nums_in_play)
            except InsufficientFundsException as e:
                print(f"Player {p} exceeded the budget")
                self.penalty(p)
            except Exception as e:
                print(f"Player {p}: {e}")
            
        return target_price, buy_price, sell_price

    def run(self):
        for i in range(self.rounds):
            target_price, buy_price, sell_price = self.run_one_round()
            if self.reshuffle:
                self.reset_cards()

            if self.display:
                self.display_money(target_price, buy_price, sell_price, i)

        winner = self.get_winner()
        if self.display:
            print("=====================================")
            print(f"Winner is: {winner}")
        
        return winner
        

    def display_money(self, target_price, buy_price, sell_price, round_number):
        print("=====================================")
        print(f"Round {round_number + 1} - target: {target_price}, buy: {buy_price}, sell: {sell_price}")
        print("Current money count: ")
        players = sorted(self.players, key=lambda x: self.money[x], reverse=True)
        for p in players:
            print(f"{p.name}: {self.money[p]}")

    def get_winner(self):
        return max(self.players, key=lambda x: self.money[x])
        

if __name__ == "__main__":
    game = Game(
        display=True, 
        rounds=random.randint(50,250),
    )
    game.setup([buildAlwaysBuy1, buildAlwaysSell1, buildAndy, buildArthur, buildWill_1, buildWill_2, buildWill_3])
    # game.setup([buildArthur, buildWill_1, buildWill_2, buildWill_3])
    winner = game.run()
