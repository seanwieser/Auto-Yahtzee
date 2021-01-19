import numpy as np

class Yahtzee:
    def __init__(self):
        self.roll = []
        self.roll_counts = []
        self.yahtzee_count = 0
        self.value_entered = False
        self.bonus = False

        self.YAHTZEE_KEYS = ['Aces', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes', \
                           '4 Kind', '3 Kind', 'Yahtzee', 'Full House', 'Small Straight', 'Large Straight', 'Chance']
        self.scoreboard = dict.fromkeys(self.YAHTZEE_KEYS, -1)
        self.potential_scores = dict()
        self.score = 0

    def update_roll(self, roll):
        self.roll = sorted(roll)
        self.roll_counts = [self.roll.count(i) for i in range(1,7)]
        self.value_entered = False
        self.update_potential_scores()

    def enter_highest_possible(self):
        for score in self.potential_score('list'):
            if self.empty_value(score[0]) or score[0]=='Yahtzee':
                self.enter_score(score)
                break

    def empty_value(self, key):
        return self.scoreboard[key] < 0

    def enter_score(self, key, value):
        if not self.value_entered:
            if key =='Yahtzee':
                self.yahtzee_count+=1
                self.scoreboard[key] += value
            else:
                self.scoreboard[key] = value
            self.value_entered = True
            self.update_score()

    def update_bonus(self):
        check = 0
        for key in self.YAHTZEE_KEYS[:6]:
            if self.scoreboard[key] >= 0:
                check += self.scoreboard[key]
                
        self.bonus = check >= 63

    def update_score(self):
        temp_score = 0
        self.update_bonus()
        for key in self.scoreboard:
            if self.scoreboard[key] >= 0:
                temp_score += self.scoreboard[key]
        if self.bonus:
            temp_score+=35
        self.score = temp_score
        
    def update_potential_scores(self):
        self.potential_scores = dict(self.potential_score('list'))
        
    def clear_potential_scores(self):
        self.potential_scores = dict()

    def combo_scores(self):
        combo_scores = self.get_n_kind()
        combo_scores.append(self.get_full_house())
        return combo_scores

    def get_yahtzee_score(self):
        if self.yahtzee_count == 0:
            return 51
        else:
            return 100

    def get_n_kind(self):
        total = np.sum(self.roll)
        if max(self.roll_counts) == 5:
            return [total, total, self.get_yahtzee_score()]
        elif max(self.roll_counts) == 4:
            return [total, total, 0]
        elif max(self.roll_counts) == 3:
            return [0, total, 0]
        else:
            return [0, 0, 0]

    def get_full_house(self):
        if (3 in self.roll_counts) and (2 in self.roll_counts):
            return 25
        else:
            return 0

    def run_scores(self):
        test_roll = [1 if i else 0 for i in self.roll_counts]
        if test_roll in [[1,1,1,1,1,0], [0,1,1,1,1,1]]:
            return [30, 40]
        elif test_roll in [[1,1,1,1,0,1], [1,0,1,1,1,1], [1,1,1,1,0,0], [0,1,1,1,1,0], [0,0,1,1,1,1]]:
            return [30, 0]
        else:
            return [0, 0]
        
    def potential_upper_score(self):
        total = np.sum(self.roll)
        return self.combo_scores() + self.run_scores() + [total]

    def potential_lower_score(self):
        lower_scores = [0,0,0,0,0,0]
        for die in self.roll:
            lower_scores[die-1] += die
        return lower_scores

    def potential_score(self, flag):
        potential_scores = dict(zip(self.YAHTZEE_KEYS, self.potential_lower_score() + self.potential_upper_score()))
        if flag == 'list':
            return sorted(potential_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            raw = str(sorted(potential_scores.items(), key=lambda x: x[1], reverse=True))
            rm_char = ['[',']',"'", '(', ')']
            for char in rm_char:
                raw = raw.replace(char, '')
            for i in range(13):
                raw = raw.replace(',', ':', 1).replace(',', ' ||', 1)
            return raw

if __name__ == "__main__":
    y = Yahtzee()
