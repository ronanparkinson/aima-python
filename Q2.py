#Q2

import random
import agents
from agents import Agent
from search import Problem, breadth_first_graph_search
from assignmentOne import GameEnvironment

class Rock(agents.Obstacle):
    pass

class Coin(agents.Thing):
    pass

class RandomAgent(Agent):
    def __init__(self):
        super().__init__()
        self.program = self.possibleMoves

    def possibleMoves(self):
        return random.choice(['left', 'right', 'up', 'down'])
    
class SearchEnvironment(GameEnvironment):

    def __init__(self, width=8, height=8):
        super().__init__(width, height)
        self.add_coins()
        self.add_Rock()

    def add_coins(self):
        return super().add_coins()
    def add_Rock(self):
        return super().add_Rock()
    
class CoinProblem(Problem):
    def __init__(self, initial_state, width, height):
        print("Initial state:", initial_state)
        self.width = width
        self.height = height
        super().__init__(initial_state)

    def moves(self, state):
        (x, y), coin_location, rocks_locations = state
        coin_location = set(coin_location)
        rocks_locations = set(rocks_locations)
        print("Current state:", state)
        print("testing new moves here!")
        moves = []
        #chat
        if x > 0 and (x - 1, y) not in rocks_locations:
            moves.append('left')
        if x < self.width - 1 and (x + 1, y) not in rocks_locations:
            moves.append('right')
        if y > 0 and (x, y - 1) not in rocks_locations:
            moves.append('down')
        if y < self.height - 1 and (x, y + 1) not in rocks_locations:
            moves.append('up')
        
        return moves
    
    def result(self, state, action):
        (x, y), coin_location, rocks_locations = state
        coin_location = set(coin_location)
        rocks_locations = set(rocks_locations)

        if action == 'left':
            destination = (x - 1, y)
            if (x, y) in coin_location:
                coin_location.remove((x, y))
            return (destination, tuple(coin_location))
        elif action == 'right':
            destination = (x + 1, y)
            if (x, y) in coin_location:
                coin_location.remove((x, y))
            return (destination, tuple(coin_location))
        elif action == 'up':
            destination = (x, y - 1)
            if (x, y) in coin_location:
                coin_location.remove((x, y))
            return (destination, tuple(coin_location))
        elif action == 'down':
            destination = (x, y + 1)
            if (x, y) in coin_location:
                coin_location.remove((x, y))
            return (destination, tuple(coin_location))
        
    def goal_test(self, state):
        _, coin_location = state
        return len(coin_location) == 0
    
    def step_cost(self, action):
        if action == 'left' |  action == 'right':
            return -5
        if action == 'up':
            return -1
        if action == 'down':
            return -2 
        if self.goal_test == True:
            return 1000
        
def get_inital_state_from_env(env):
    agent_location = None
    coin_location = set()
    rocks_locations = set()

    for thing in env.things:
        if isinstance(thing, Coin):
            coin_location.add(thing.location)
        elif isinstance(thing, Rock):
            rocks_locations.add(thing.location)
        elif isinstance(thing, Agent):
            agent_location = thing.location

    return(agent_location, tuple(coin_location), tuple(rocks_locations))
    
if __name__ == 'main':
    agent = RandomAgent()

    env = SearchEnvironment(width=8, height=8)
    env.add_thing(agent, (0, 0))

    initial_state = get_inital_state_from_env(env)
    print("Initial state:", initial_state)  # For debugging
    
    coin_problem = CoinProblem(initial_state, env.width, env.height)

    goal_node = breadth_first_graph_search(coin_problem)

    if goal_node:
        print("Solution found!")
        print("Solution steps:", goal_node.solution())
        print("Path to solution:", [node.state for node in goal_node.path()])
        print("Total cost:", goal_node.path_cost)
else:
    print("No solution found.")