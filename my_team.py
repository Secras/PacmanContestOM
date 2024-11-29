# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
import time, math
from util import manhattan_distance


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [DefensiveAgent(first_index), OffensiveReflexAgent(second_index)]

########################################
# Markus Part #

class SearchProblem:
   
    def get_start_state(self):
        util.raise_not_defined()

    def is_goal_state(self, state):
        util.raise_not_defined()

    def get_successors(self, state):
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        util.raise_not_defined()

class PositionSearchProblem(SearchProblem):

    def __init__(self, game_state, cost_fn = lambda x: 1, goal=(1, 1), start=(0,0), warn=True, visualize=False):
        """
        Stores the start and goal.

        game_state: A GameState obj (pacman.py)
        cost_fn: A function from a search state (tuple) to a non-negative number
        goal: A position in the game_state
        """
        self.walls = game_state.get_walls()
        self.startState = start
        self.goal = goal
        self.cost_fn = cost_fn

        
    def get_start_state(self):
        return self.startState

    def is_goal_state(self, state):
        is_goal = state == self.goal

        return is_goal

    def get_successors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        directions = {Directions.NORTH: (0, 1),
                   Directions.SOUTH: (0, -1),
                   Directions.EAST:  (1, 0),
                   Directions.WEST:  (-1, 0),
                   Directions.STOP:  (0, 0)}
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = directions[action]
            next_x, next_y = int(x + dx), int(y + dy)
            if not self.walls[next_x][next_y]:
                next_state = (next_x, next_y)
                cost = self.cost_fn(next_state)
                successors.append( ( next_state, action, cost) )

        return successors

    def get_cost_of_actions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions is None: return 999999
        x,y= self.get_start_state()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.cost_fn((x, y))
        return cost

class SearchNode:
    def __init__(self, parent, node_info):
        """
            parent: parent SearchNode.

            node_info: tuple with three elements => (coord, action, cost)

            coord: (x,y) coordinates of the node position

            action: Direction of movement required to reach node from
            parent node. Possible values are defined by class Directions from
            game.py

            cost: cost of reaching this node from the starting node.
        """

        self.__state = node_info[0]
        self.action = node_info[1]
        self.cost = node_info[2] if parent is None else node_info[2] + parent.cost
        self.parent = parent

    # The coordinates of a node cannot be modified, se we just define a getter.
    # This allows the class to be hashable.
    @property
    def state(self):
        return self.__state

    def get_path(self):
        path = []
        current_node = self
        while current_node.parent is not None:
            path.append(current_node.action)
            current_node = current_node.parent
        path.reverse()
        return path
    
    # Consider 2 nodes to be equal if their coordinates are equal (regardless of everything else)
    # def __eq__(self, __o: obj) -> bool:
    #     if (type(__o) is SearchNode):
    #         return self.__state == __o.__state
    #     return False

    # # def __hash__(self) -> int:
    # #     return hash(self.__state)

def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"
    start = SearchNode(None, (problem.get_start_state(), 'Stop', 0))
    stack = util.Stack()
    visited = set()

    stack.push(start)

    while not stack.is_empty():
        current_node = stack.pop()
        if problem.is_goal_state(current_node.state):  # Reached the goal so return the path
            return current_node.get_path()
        if current_node.state not in visited:
            visited.add(current_node.state)
            for successor in problem.get_successors(current_node.state):
                stack.push(SearchNode(current_node, successor))

def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start = SearchNode(None, (problem.get_start_state(), None, 0))
    queue = util.Queue()
    visited = set()

    queue.push(start)

    while not queue.is_empty():
        current_node = queue.pop()
        if problem.is_goal_state(current_node.state):
            return current_node.get_path()
        if current_node.state not in visited:
            visited.add(current_node.state)
            for successor in problem.get_successors(current_node.state):
                queue.push(SearchNode(current_node, successor))

def uniform_cost_search(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start = SearchNode(None, (problem.get_start_state(), None, 0))
    queue = util.PriorityQueue()
    visited = set()

    queue.push(start, 0)

    while not queue.is_empty():
        current_node = queue.pop()
        if problem.is_goal_state(current_node.state):
            return current_node.get_path()
        if current_node.state not in visited:
            visited.add(current_node.state)

            for successor in problem.get_successors(current_node.state):
                successor_node = SearchNode(current_node, successor)

                path = successor_node.get_path()
                cost = problem.get_cost_of_actions(path)
                queue.update(successor_node, cost)

def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def a_star_search(problem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start = SearchNode(None, (problem.get_start_state(), None, 0))
    queue = util.PriorityQueue()
    visited = set()

    queue.push(start, 0)

    while not queue.is_empty():
        current_node = queue.pop()
        if problem.is_goal_state(current_node.state):
            return current_node.get_path()
        if current_node.state not in visited:
            visited.add(current_node.state)

            for successor in problem.get_successors(current_node.state):
                successor_node = SearchNode(current_node, successor)

                path = successor_node.get_path()
                cost = problem.get_cost_of_actions(path) + heuristic(successor_node.state,problem)
                queue.update(successor_node, cost)

class MinState():
    
    def __init__(self, grid, positions):
        # self.grid = self.transform_grid(grid)
        self.grid = grid
        self.positions = positions
        
    def to_bool(self, string):
        if string == 'T':
            return True
        elif string == 'F':
            return False
        
    def transform_grid(self, grid):
        return [self.to_bool(cell) for row in grid.split("\n") for cell in row.split()]
        
    def generate_successor(self, agent_index, action):
        if action == 'North':
            self.positions[agent_index] = (self.positions[agent_index][0], self.positions[agent_index][1]+1)
        elif action == 'East':
            self.positions[agent_index] = (self.positions[agent_index][0]+1, self.positions[agent_index][1])
        elif action == 'South':
            self.positions[agent_index] = (self.positions[agent_index][0], self.positions[agent_index][1]-1)
        elif action == 'West':
            self.positions[agent_index] = (self.positions[agent_index][0]-1, self.positions[agent_index][1])
            
        return self
        
        
    def get_legal_actions(self, agent_index):
        directions = []
        pos_x = int(self.positions[agent_index][0])
        pos_y = int(self.positions[agent_index][1])
        # if not len(self.grid[pos_x]) == pos_y and not self.grid[pos_x][pos_y+1]:
        #     directions.append('North')
        # elif not len(self.grid) == pos_x and not self.grid[pos_x+1][pos_y]:
        #     directions.append('East')
        # elif not pos_y == 0 and not self.grid[pos_x][pos_y-1]:
        #     directions.append('South')
        # elif not pos_x == 0 and not self.grid[pos_x-1][pos_y]:
        #     directions.append('West')
        if not self.grid[pos_x][pos_y+1]:
            directions.append('North')
        elif not self.grid[pos_x+1][pos_y]:
            directions.append('East')
        elif not self.grid[pos_x][pos_y-1]:
            directions.append('South')
        elif not self.grid[pos_x-1][pos_y]:
            directions.append('West')
        
        return directions

class DefensiveAgent(CaptureAgent):
    
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.features = {}
        self.max_depth = 5
    
    
    def choose_action(self, game_state):
        
        food_eaten = self.compare_foods(game_state)
        self.features['prev_food'] = set(self.get_food_you_are_defending(game_state).as_list())
        
        my_pos = game_state.get_agent_state(self.index).get_position()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        if len(invaders) > 0:
            invaders.sort(key= lambda x: self.get_maze_distance(my_pos, x.get_position()))
            self.features['dest'] = invaders[0].get_position()
        if len(invaders) == 1:
            # print("-----------------------------")
            # I see one enemy
            scared = game_state.get_agent_state(self.index).scared_timer > 0
            if util.manhattan_distance(my_pos, invaders[0].get_position()) == 1 and not scared:
                return self.get_direction_enemy(my_pos, invaders[0].get_position())
            legal_actions = game_state.get_legal_actions(0)
            max_action = 'Stop'
            max_value = float('-inf')
            for action in legal_actions:
                enemy_pos = invaders[0].get_position()
                new_pos = self.get_successor(game_state, action).get_agent_state(self.index).get_position()
                successor = MinState(game_state.get_walls(), 
                                   [new_pos, enemy_pos])
                minimax_value = self.minimax(successor, 1, 0, self.max_depth)
                # print("---")
                # print(minimax_value)
                # print(action)
                # print("---")
                if  minimax_value > max_value:
                    max_value = minimax_value
                    max_action = action
            # print(max_action)
            # time.sleep(5)
            return max_action
        elif len(invaders) == 2:
            # I see two enemies
            invaders.sort(key= lambda x: self.get_maze_distance(my_pos, x.get_position()))
            nearest_enemy = invaders[0].get_position()
            legal_actions = game_state.get_legal_actions(0)
            max_action = 'Stop'
            max_value = float('-inf')
            for action in legal_actions:
                enemy_pos = nearest_enemy.get_position()
                new_pos = self.get_successor(game_state, action).get_agent_state(self.index).get_position()
                successor = MinState(game_state.get_walls(), 
                                   [new_pos, enemy_pos])
                minimax_value = self.minimax(successor, 1, 0, self.max_depth)
                # print("---")
                # print(minimax_value)
                # print(action)
                # print("---")
                if  minimax_value > max_value:
                    max_value = minimax_value
                    max_action = action
            return max_action
        else:
            # I don't see an enemy 
            if food_eaten:
                # food was eaten => go to the crime scene
                self.features['dest'] = food_eaten
            if self.features.get('dest', my_pos) == my_pos:
                # I have a previous goal
                do = True
                next_dest = None
                while do or next_dest == my_pos:
                    do = False
                    # next_dest = self.pick_next_food(game_state)
                    next_dest = random.choice(self.get_food_you_are_defending(game_state).as_list())
                self.features['dest'] = next_dest
            path = breadth_first_search(PositionSearchProblem(game_state, 
                                                              goal=self.features['dest'], 
                                                              start=my_pos, 
                                                              visualize=False))
            if path == None:
                self.features['dest'] = my_pos
                return 'Stop'
            return path[0]
        
    def get_direction_enemy(self, my_pos, enemy_pos):
        if my_pos[1] == enemy_pos[1]:
            if my_pos[0] < enemy_pos[0]:
                return 'East'
            elif my_pos[0] > enemy_pos[0]:
                return 'West'
            else:
                return 'Stop'
        elif my_pos[0] == enemy_pos[0]:
            if my_pos[1] < enemy_pos[1]:
                return 'North'
            elif my_pos[1] > enemy_pos[1]:
                return 'South'
            else:
                return 'Stop'
        else:
            return 'Stop'
        
    def pick_next_food(self, game_state):
        """Pick the next food to patrol to."""
        distances = {}
        my_pos = game_state.get_agent_state(self.index).get_position()
        for food in self.get_food_you_are_defending(game_state).as_list():
            distances[food] = min([abs(util.manhattan_distance(my_pos, food) - game_state.get_agent_distances()[enemy_ind]) 
                                  for enemy_ind in self.get_opponents(game_state)])
        min_dist = float('+inf')
        next_food = (0,0)
        for food, distance in distances.items():
            if distance < min_dist:
                next_food = food
                min_dist = distance
        return next_food
        
    def minimax(self, min_state: MinState, agent_index, current_depth, max_depth):
        """minimax for two agents: agent 0 is max, agent 1 is min

        Args:
            game_state (_type_): _description_
            current_depth (_type_): _description_
            agent_index (_type_): _description_

        Returns:
            _type_: _description_
        """
        next_depth = current_depth
        next_agent = agent_index
        if agent_index == 1:
            # last eval for this depth => next depth in next call
            next_agent = 0
            next_depth += 1
        else:
            next_agent += 1
        if current_depth == max_depth:
            return self.evaluation_function(min_state)
        legal_actions = min_state.get_legal_actions(agent_index)
        action_evals = [self.minimax(min_state.generate_successor(agent_index, action), 
                                     next_agent, 
                                     next_depth,
                                     max_depth) for action in legal_actions]
        if agent_index == self.index:
            # pacman moves (MAX)
            return max(action_evals)
        else:
            # one of the ghosts moves (MIN)
            return min(action_evals)
        
    def evaluation_function(self, min_state):
        # print((-1) * util.manhattan_distance(min_state.positions[0], min_state.positions[1]))
        return (-1) * self.get_maze_distance(min_state.positions[0], min_state.positions[1])
        # return (-1) * util.manhattan_distance(min_state.positions[0], min_state.positions[1])
        
    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor
        
    def compare_foods(self, game_state):
        current_food = set(self.get_food_you_are_defending(game_state).as_list())
        diff = self.features.get('prev_food', current_food) - current_food
        if diff != set():
            # some food was eaten
            return diff.pop()
        else:
            return False

##############################################
# Oliver's Part#

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.from_start = 0

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        action_to_take = random.choice(best_actions)
        is_red = self.red
        # print('Is red: ', is_red)
        if (is_red and action_to_take == Directions.EAST) or (not is_red and action_to_take == Directions.WEST):
            self.from_start += 1

        # print("Action to take: ", action_to_take)
        # print("From start: ", self.from_start)
        return action_to_take

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.food_eaten = 0
        self.chosen_food = None
        self.countdown = 3
        self.target_cluster_index = 0

        self.positions_record = []
        self.record_threshold = 10
        self.agent_stuck = False
        self.cooldown = 0  # Cooldown period to avoid frequent analysis
        self.cooldown_threshold = 10  # Number of steps to wait before re-analysis

        # Capsule
        self.capsule_active = False
        self.capsule_cooldown = 40

    def update_recent_positions(self, position):
        """Update the record of recent positions."""
        self.positions_record.append(position)
        if len(self.positions_record) > self.record_threshold:
            self.positions_record.pop(0)  # Maintain a fixed size

    def analyse_movement(self):
        """Analyse the agent's movement to detect patterns. Goal is to avoid getting stuck in a loop when by the edge
            and waiting for the ghost to go away.
        """

        if self.cooldown > 0:
            self.cooldown -= 1
            return

        if len(self.positions_record) < self.record_threshold:
            return

        if len(set(self.positions_record)) < 3:
            self.agent_stuck = True
            self.positions_record = []
            return

        self.agent_stuck = False

    def minimax_evaluation_function(self, game_state, agent_index):
        """Evaluation function for the MiniMax algorithm. Max player (PacMan) tries to avoid the ghost and return to base.
            We predict that the Min player (Ghost) will try to chase PacMan.
        """

        agent_state = game_state.get_agent_state(agent_index)
        agent_pos = agent_state.get_position()

        if agent_index == self.index:  # PacMan's perspective
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            ghost_positions = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
            ghost_distances = [self.get_maze_distance(agent_pos, g.get_position()) for g in ghost_positions]

            # Base position
            base_pos = self.start
            base_distance = self.get_maze_distance(agent_pos, base_pos)

            # Avoid ghosts
            min_ghost_distance = min(ghost_distances) if ghost_distances else float('inf')
            ghost_penalty = -500 if min_ghost_distance <= 1 else -100 / max(1, min_ghost_distance)

            # Incentivize reaching base
            base_reward = 200 / (1 + base_distance)
            if self.food_eaten <= 2:
                base_reward = 0  # No need to return to base if food eaten is less than 2

            evaluation = ghost_penalty + base_reward
            # print(f"[Evaluation - PacMan] Position: {agent_pos}, Base Distance: {base_distance}, "
            #       f"Min Ghost Distance: {min_ghost_distance}, Evaluation: {evaluation}")
            return evaluation
        else:  # Ghost's perspective
            pacman = game_state.get_agent_state(self.index)
            pacman_pos = pacman.get_position()
            pacman_distance = self.get_maze_distance(agent_pos, pacman_pos) if pacman_pos else float('inf')

            # Incentivize chasing PacMan
            chase_reward = 100 / max(1, pacman_distance)

            # Discourage staying idle (if the ghost is not near PacMan)
            idle_penalty = -10 if pacman_distance > 5 else 0

            evaluation = chase_reward + idle_penalty
            # print(f"[Evaluation - Ghost] Position: {agent_pos}, PacMan Distance: {pacman_distance}, "
            #       f"Evaluation: {evaluation}")
            return evaluation

    def minimax(self, game_state, current_depth, agent_index):
        """MiniMax algorithm to predict the best move for PacMan and the ghosts."""

        agents = [self.index]
        for enemy in self.get_opponents(game_state):
            enemy_state = game_state.get_agent_state(enemy)
            if not enemy_state.is_pacman and enemy_state.get_position() is not None:
                agents.append(enemy)
                break

        next_depth = current_depth
        next_agent = agent_index

        if agent_index == len(agents) - 1:
            # last eval for this depth => next depth in next call
            next_agent = 0
            next_depth += 1
        else:
            next_agent += 1

        # print(f"Agent: {agent_index}, Depth: {current_depth}, Next Depth: {next_depth}, Next Agent: {next_agent}")

        if current_depth == 2:
            return self.minimax_evaluation_function(game_state, agents[agent_index])

        # If ghost is no longer visible, return a high evaluation
        if agent_index > 0 and agent_index > len(agents) - 1:
            return 1000

        actions = game_state.get_legal_actions(agents[agent_index])
        action_evals = [self.minimax(game_state.generate_successor(agents[agent_index], action), next_depth, next_agent)
                        for action in actions]

        if agent_index == 0:
            # pacman moves (MAX)
            return max(action_evals)
        else:
            # one of the ghosts moves (MIN)
            return min(action_evals)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # Analyse the agent's movement
        curr_pos = game_state.get_agent_position(self.index)
        self.update_recent_positions(curr_pos)
        self.analyse_movement()

        is_pacman = game_state.get_agent_state(self.index).is_pacman
        is_stuck = False
        if self.agent_stuck:
            print('Agent is stuck.')
            is_stuck = True

        actions = game_state.get_legal_actions(self.index)
        start = time.time()

        # Run MiniMax if ghost is nearby
        ghost_distance = self.ghost_distance(game_state)
        if ghost_distance > 0 and not self.capsule_active and not is_stuck:
            max_action = max(actions,
                             key=lambda action: self.minimax(game_state.generate_successor(self.index, action), 0, 0))
            # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
            return max_action

        # Otherwise run the normal evaluation
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        action_to_take = random.choice(best_actions)
        successor = self.get_successor(game_state, action_to_take)
        next_pos = successor.get_agent_state(self.index).get_position()

        # Check if the agent has eaten food
        if next_pos in self.get_food(game_state).as_list():
            self.food_eaten += 1

        if not game_state.get_agent_state(self.index).is_pacman:
            self.food_eaten = 0

        # Capsule Activation / Deactivation
        if self.capsule_active:
            self.capsule_cooldown -= 1
            if self.capsule_cooldown == 0:
                self.capsule_active = False
                self.capsule_cooldown = 40

        if game_state.get_agent_state(self.index).is_pacman and next_pos in self.get_capsules(game_state):
            self.capsule_active = True

        return action_to_take

    def form_food_clusters(self, food_list: list, distance_threshold: int) -> list:
        """Form clusters of food items based on the distance threshold.
        :param food_list: List of food items
        :param distance_threshold: Maximum distance between food items to be considered in the same cluster
        :return: List of clusters of food items
        """

        clusters = []

        for food in food_list:
            added_to_cluster = False

            # Try to add the food to an existing cluster
            for cluster in clusters:
                if any(manhattan_distance(food, existing_food) <= distance_threshold for existing_food in cluster):
                    cluster.append(food)
                    added_to_cluster = True
                    break

            # If no cluster is close enough, create a new cluster
            if not added_to_cluster:
                clusters.append([food])

        return clusters

    def rank_clusters(self, clusters: list, my_pos: tuple, is_red_team: bool, w1=1, w2=1, w3=1) -> list:
        """Rank the clusters based on the distance to the agent, cluster size, and proximity to the desired edge.
        :param clusters: List of clusters of food items
        :param my_pos: Position of the agent
        :param is_red_team: True if the agent is in the red team
        :param w1: Weight for the distance from the agent
        :param w2: Weight for the cluster size
        :param w3: Weight for the proximity to the red/blue team boundary
        :return: List of ranked clusters
        """

        ranked_clusters = []

        for cluster in clusters:
            # Calculate distances for all food items in the cluster
            distances = [self.get_maze_distance(my_pos, food) for food in cluster]
            min_distance = min(distances)
            cluster_size = len(cluster)

            # Calculate the X-priority score (proximity to the desired edge)
            if is_red_team:
                # Red team prioritizes lower X-values
                edge_proximity_score = sum(-food[0] for food in cluster) / cluster_size
            else:
                # Blue team prioritizes higher X-values
                edge_proximity_score = sum(food[0] for food in cluster) / cluster_size

            # Compute weighted score
            score = (w1 / min_distance) + (w2 * cluster_size) + (w3 * edge_proximity_score)

            cluster.sort(key=lambda f: manhattan_distance(my_pos, f))

            # Add cluster details
            ranked_clusters.append({
                "score": score,
                "cluster": cluster,
                "min_distance": min_distance,
                "size": cluster_size,
                "edge_proximity_score": edge_proximity_score
            })

        # Sort clusters by score
        ranked_clusters.sort(key=lambda x: -x["score"])

        return ranked_clusters

    def ghost_distance(self, successor):
        """Calculates the distance to the nearest ghost. If no ghost is visible, returns 0."""

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        if len(invaders) > 0:
            dists = [self.get_maze_distance(successor.get_agent_state(self.index).get_position(), a.get_position()) for
                     a in invaders]
            return min(dists)
        return 0

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        next_pos = successor.get_agent_state(self.index).get_position()
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Compute distance to the nearest food item
        food_list = self.get_food(successor).as_list()
        food_clusters = self.form_food_clusters(food_list, 3)

        if self.capsule_active:  # Prioritize bigger food clusters and ignore the rest
            ranked_clusters = self.rank_clusters(food_clusters, successor.get_agent_state(self.index).get_position(),
                                                 self.red, w1=1, w2=2, w3=0)
        else:
            ranked_clusters = self.rank_clusters(food_clusters, successor.get_agent_state(self.index).get_position(),
                                                 self.red)

        features['successor_score'] = -len(food_list)

        # Compute distance to the nearest food item
        if len(food_list) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            
            if self.chosen_food in game_state.get_walls().as_list():
                self.chosen_food = None

            if self.countdown == 0 or self.chosen_food is None:
                nearest_food = ranked_clusters[0]["cluster"][0]
                self.chosen_food = nearest_food
                self.countdown = 5
            else:
                self.countdown -= 1

            features['distance_to_target'] = self.get_maze_distance(my_pos, self.chosen_food)

        if self.food_eaten > 3 and not self.capsule_active:
            features['distance_to_target'] = self.get_maze_distance(my_pos, self.start)

        if self.agent_stuck:
            features['return_to_start'] = self.get_maze_distance(my_pos, self.start)

        return features

    def get_weights(self, game_state, action):
        return {'find_more_food': 0,
                'successor_score': 100,
                'distance_to_target': -1,
                'return_to_start': -1000}

# class DefensiveReflexAgent(ReflexCaptureAgent):
#     """
#     A reflex agent that keeps its side Pacman-free. Again,
#     this is to give you an idea of what a defensive agent
#     could be like.  It is not the best or only way to make
#     such an agent.
#     """

#     def get_features(self, game_state, action):
#         features = util.Counter()
#         successor = self.get_successor(game_state, action)
#         print(self.index)

#         my_state = successor.get_agent_state(self.index)
#         my_pos = my_state.get_position()

#         # Computes whether we're on defense (1) or offense (0)
#         print("Is PacMan", my_state.is_pacman)
#         features['on_defense'] = 1
#         if my_state.is_pacman: features['on_defense'] = 0

#         # Computes distance to invaders we can see
#         enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
#         invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
#         features['num_invaders'] = len(invaders)
#         if len(invaders) > 0:
#             dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
#             features['invader_distance'] = min(dists)

#         if action == Directions.STOP: features['stop'] = 1
#         rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
#         if action == rev: features['reverse'] = 1

#         return features

#     def get_weights(self, game_state, action):
#         return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
