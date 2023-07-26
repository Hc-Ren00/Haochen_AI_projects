import random
import math

BOT_NAME = "BUTT KICKER"


class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""

    def __init__(self, sd=None):
        if sd is None:
            self.st = None
        else:
            random.seed(sd)
            self.st = random.getstate()

    def get_move(self, state):
        if self.st is not None:
            random.setstate(self.st)
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move."""

    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""

    def get_move(self, state):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player()
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            util = self.minimax(state)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state

    def maxValue(self, state):
        if state.is_full():
            return state.utility()
        v = -math.inf
        for s in state.successors():
            v = max(v, self.minValue(s[1]))
        return v

    def minValue(self, state):
        if state.is_full():
            return state.utility()
        v = math.inf
        for s in state.successors():
            v = min(v, self.maxValue(s[1]))
        return v

    def minimax(self, state):
        """Determine the minimax utility value of the given state.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the exact minimax utility value of the state
        """
        result = 0
        if state.next_player() == 1:
            result = self.minValue(state)
        else:
            result = self.maxValue(state)
        return result


class MinimaxHeuristicAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def maxValued(self, state, depth):
        if depth is None:
            depth = -1
        if depth == 0:
            return self.evaluation(state)
        if state.is_full():
            return state.utility()
        v = -math.inf
        for s in state.successors():
            v = max(v, self.minValued(s[1], depth - 1))
        return v

    def minValued(self, state, depth):
        if depth is None:
            depth = -1
        if depth == 0:
            return self.evaluation(state)
        if state.is_full():
            return state.utility()
        v = math.inf
        for s in state.successors():
            v = min(v, self.maxValued(s[1], depth - 1))
        return v

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state.
        The depth data member (set in the constructor) determines the maximum depth of the game
        tree that gets explored before estimating the state utilities using the evaluation()
        function.  If depth is 0, no traversal is performed, and minimax returns the results of
        a call to evaluation().  If depth is None, the entire game tree is traversed.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        d = self.depth_limit
        if d == 0:
            return self.evaluation(state)
        result = 0
        if state.next_player() == 1:
            result = self.minValued(state, d)
        else:
            result = self.maxValued(state, d)
        return result


    def eva(self, state, str):
        """Give an evaluation of the current state to help AI figuring out a better move. This
        algorithm considers the points you already have and also the points that you
        can potentially get. Besides, it also takes into account how many points your "enemy"
        is going to get and will optimally choose the best move.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the score of this state
        """
        keys = []
        if str == "rows":
            keys = state.get_rows()
        elif str == "cols":
            keys = state.get_cols()
        elif str == "diags":
            keys = state.get_diags()
        score = 0
        for r in keys:
            if 1 in r:
                right = -1
                index = r.index(1, right + 1,)
                while type(index) == int:
                    curscore = 0
                    left = index
                    right = index
                    while right + 1 < len(r):
                        if r[right + 1] == 1:
                            right = right + 1
                        else:
                            break
                    if right - left >= 2:
                        curscore += pow(right - left + 1, 2)
                    if left - 1 >= 0 and right + 1 < len(r):
                        if r[left - 1] == 0 and r[right + 1] == 0 and right - left >= 1:
                            curscore += 2
                        elif r[left - 1] == 0 or r[right + 1] == 0 and right - left >= 1:
                            curscore += 1
                    elif left - 1 >= 0:
                        if r[left - 1] == 0 and right - left >= 1:
                            curscore += 1
                    elif right + 1 < len(r):
                        if r[right + 1] == 0 and right - left >= 1:
                            curscore += 1

                    score += curscore
                    if right + 1 >= len(r):
                        break
                    else:
                        list = []
                        for i in range(right + 1, len(r) - 1):
                            list.append(r[i])
                        if 1 in list:
                            index = r.index(1, right + 1,)
                        else:
                            break

            if -1 in r:
                right = -1
                index = r.index(-1, right + 1,)
                while type(index) == int:
                    curscore = 0
                    left = index
                    right = index
                    while right + 1 < len(r):
                        if r[right + 1] == -1:
                            right = right + 1
                        else:
                            break
                    if right - left >= 2:
                        curscore -= pow(right - left + 1, 2)
                    if left - 1 >= 0 and right + 1 < len(r):
                        if r[left - 1] == 0 and r[right + 1] == 0 and right - left >= 1:
                            curscore -= 2
                        elif r[left - 1] == 0 or r[right + 1] == 0 and right - left >= 1:
                            curscore -= 1
                    elif left - 1 >= 0:
                        if r[left - 1] == 0 and right - left >= 1:
                            curscore -= 1
                    elif right + 1 < len(r):
                        if r[right + 1] == 0 and right - left >= 1:
                            curscore -= 1

                    score += curscore
                    if right + 1 >= len(r):
                        break
                    else:
                        list = []
                        for i in range(right + 1, len(r) - 1):
                            list.append(r[i])
                        if -1 in list:
                            index = r.index(-1, right + 1, )
                        else:
                            break

        return score

    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        N.B.: This method must run in constant time for all states!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heuristic estimate of the utility value of the state
        """

        return self.eva(state, "rows") + self.eva(state, "cols") + self.eva(state, "diags")


class MinimaxPruneAgent(MinimaxAgent):
    """Smarter computer agent that uses minimax with alpha-beta pruning to select the best move."""

    def maxValuep(self, state, alpha, beta):
        if state.is_full():
            return state.utility()
        v = -math.inf
        for s in state.successors():
            v = max(v, self.minValuep(s[1], alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def minValuep(self, state, alpha, beta):
        if state.is_full():
            return state.utility()
        v = math.inf
        for s in state.successors():
            v = min(v, self.maxValuep(s[1], alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def minimax(self, state):
        """Determine the minimax utility value the given state using alpha-beta pruning.

        The value should be equal to the one determined by MinimaxAgent.minimax(), but the
        algorithm should do less work.  You can check this by inspecting the value of the class
        variable GameState.state_count, which keeps track of how many GameState objects have been
        created over time.  This agent does not use a depth limit like MinimaxHeuristicAgent.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors().  That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to to column 1.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """

        result = 0
        if state.next_player() == 1:
            result = self.minValuep(state, -math.inf, math.inf)
        else:
            result = self.maxValuep(state, -math.inf, math.inf)
        return result
