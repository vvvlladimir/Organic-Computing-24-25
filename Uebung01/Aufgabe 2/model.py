from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from agents import AntAgent, ObjectAgent
import random

class ClusteringModel(Model):
    def __init__(self, width, height, num_agents=20, num_objects=200):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)

        # Parameters for agents
        self.PICKUP_THRESHOLD = 0.1
        self.DROP_THRESHOLD = 0.3
        self.ALPHA = 0.5
        self.SIGMA_SQUARED = 25

        # Object creation
        for _ in range(num_objects):
            object_type = random.choice(range(3))  # three types of objects, e.g. 0, 1 and 2
            obj = ObjectAgent(self, object_type)
            self.grid.place_agent(obj, (random.randrange(self.grid.width), random.randrange(self.grid.height)))

        # Creating ants
        for _ in range(num_agents):
            ant = AntAgent(self)
            self.grid.place_agent(ant, (random.randrange(self.grid.width), random.randrange(self.grid.height)))
            self.schedule.add(ant)

    def step(self):
        self.schedule.step()