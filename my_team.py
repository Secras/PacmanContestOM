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
from contest.util import manhattan_distance


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
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

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
        print('Is red: ', is_red)
        if (is_red and action_to_take == Directions.EAST) or (not is_red and action_to_take == Directions.WEST):
            self.from_start += 1

        print("Action to take: ", action_to_take)
        print("From start: ", self.from_start)
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
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
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

    def update_recent_positions(self, position):
        # Update the list of recent positions
        self.positions_record.append(position)
        if len(self.positions_record) > self.record_threshold:
            self.positions_record.pop(0)  # Maintain a fixed size

    def analyse_movement(self):
        if self.cooldown > 0:
            self.cooldown -= 1
            return

        if len(self.positions_record) < self.record_threshold:
            return

        # Check if the agent is moving in the same four locations
        if len(set(self.positions_record[-4:])) == 1:
            self.agent_stuck = True
            return

        # Check for looping patterns (e.g., same sequence of moves)
        for i in range(1, len(self.positions_record) // 2 + 1):
            if self.positions_record[-i:] == self.positions_record[-2 * i:-i]:
                self.agent_stuck = True
                return

        self.agent_stuck = False

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # Analyse the agent's movement
        curr_pos = game_state.get_agent_position(self.index)
        self.update_recent_positions(curr_pos)
        self.analyse_movement()

        is_pacman = game_state.get_agent_state(self.index).is_pacman
        if self.agent_stuck and not is_pacman and self.cooldown == 0:
            print('Agent is stuck.')
            self.target_cluster_index = (self.target_cluster_index + 1) % 3
            print('Target cluster index:', self.target_cluster_index)
            self.cooldown = self.cooldown_threshold

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

        return action_to_take

    def form_food_clusters(self, food_list, distance_threshold):
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

    def rank_clusters(self, clusters, my_pos, is_red_team, w1=1, w2=1, w3=1):
        ranked_clusters = []

        for cluster in clusters:
            # Calculate distances for all food items in the cluster
            distances = [self.get_maze_distance(my_pos, food) for food in cluster]
            min_distance = min(distances)
            cluster_size = len(cluster)
            edge_proximity_score = 0

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

    def ghost_threat(self, successor):
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        if len(invaders) > 0:
            dists = [self.get_maze_distance(successor.get_agent_state(self.index).get_position(), a.get_position()) for a in invaders]
            return 1.0 / min(dists)  # Higher threat as the ghost gets closer
        return 0

    def ghost_distance(self, successor):
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        if len(invaders) > 0:
            dists = [self.get_maze_distance(successor.get_agent_state(self.index).get_position(), a.get_position()) for a in invaders]
            return min(dists)
        return 0

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        next_pos = successor.get_agent_state(self.index).get_position()
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Check if the agent has eaten food
        if next_pos in self.get_food(game_state).as_list():
            self.food_eaten += 1

        if not game_state.get_agent_state(self.index).is_pacman:
            self.food_eaten = 0

        # Compute distance to the nearest food item
        food_list = self.get_food(successor).as_list()
        food_clusters = self.form_food_clusters(food_list, 3)
        ranked_clusters = self.rank_clusters(food_clusters, successor.get_agent_state(self.index).get_position(), self.red)

        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest ghost
        features['ghost_threat'] = self.ghost_threat(successor)

        # Penalize actions that move closer to the ghost
        features['distance_to_ghost'] = self.ghost_distance(successor)

        # Compute distance to the nearest food item
        if len(food_list) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()

            if self.countdown == 0 or self.chosen_food is None:
                nearest_food = ranked_clusters[self.target_cluster_index]["cluster"][0]
                self.chosen_food = nearest_food
                self.countdown = 3
            else:
                self.countdown -= 1

            # print('Nearest food:', self.chosen_food)
            features['distance_to_food'] = self.get_maze_distance(my_pos, self.chosen_food)
            # print('Distance to food:', features['distance_to_food'])

        if self.food_eaten > 3:
            features['distance_to_food'] = self.get_maze_distance(my_pos, self.start)

        # Add feature to return to start if ghost is nearby and food eaten is at least 2
        if features['ghost_threat'] > 0 and self.food_eaten >= 2:
            features['return_to_start'] = self.get_maze_distance(my_pos, self.start)
            features['distance_to_food'] = 0
            features['successor_score'] = 0

        return features

    def get_weights(self, game_state, action):
        return {'find_more_food': 0,
                'successor_score': 100,
                'distance_to_food': -1,
                'ghost_threat': 1000,
                'distance_to_ghost': 1000,
                'return_to_start': -1000}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        print(self.index)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        print("Is PacMan", my_state.is_pacman)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}