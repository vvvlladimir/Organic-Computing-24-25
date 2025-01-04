import numpy as np
import random
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from agents import AntAgent, ObjectAgent

class ClusteringModel(Model):
    def __init__(self, width, height, num_agents=20, num_objects=200):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        self._initialize_grid(num_objects, num_agents)
        self.start_entropies = {}  # Store initial entropies for all agents

        # Data Collection for each attribute
        self.datacollector = DataCollector(
            model_reporters={
                "Ant_Emergence_X": lambda m: m.calculate_emergence('x_position', AntAgent),
                "Ant_Emergence_Y": lambda m: m.calculate_emergence('y_position', AntAgent),
                "Ant_Emergence_Particle": lambda m: m.calculate_emergence('particle_carried', AntAgent),
                "Ant_Average_Entropy_X": lambda m: m.average_entropy('x_position', AntAgent),
                "Ant_Average_Entropy_Y": lambda m: m.average_entropy('y_position', AntAgent),
                "Ant_Average_Entropy_Particle": lambda m: m.average_entropy('particle_carried', AntAgent),

                "OBJ_Average_Entropy_X": lambda m: m.average_entropy('x_position', ObjectAgent),
                "OBJ_Average_Entropy_Y": lambda m: m.average_entropy('y_position', ObjectAgent),
                "OBJ_Average_Entropy_Neighbors": lambda m: m.average_entropy('neighbors', ObjectAgent),
            }
        )

    def _initialize_grid(self, num_objects, num_agents):
        """Place objects and agents on the grid."""
        for _ in range(num_objects):
            object_type = random.randint(0, 2)
            x, y = self._random_empty_cell()
            obj = ObjectAgent(self, object_type)
            self.grid.place_agent(obj, (x, y))
            self.schedule.add(obj)  # Add to scheduler


        for _ in range(num_agents):
                x, y = self._random_empty_cell()
                ant = AntAgent(self)
                self.grid.place_agent(ant, (x, y))
                self.schedule.add(ant)

    def _random_empty_cell(self):
        """Find a random empty cell."""
        while True:
            x, y = random.randrange(self.grid.width), random.randrange(self.grid.height)
            if self.grid.is_cell_empty((x, y)):
                return x, y

    def calculate_emergence(self, attribute, type_filter):
        """Calculate emergence for a specific attribute, optionally filtered by agent type."""
        # Get the current entropy values as a flat dictionary
        current_entropies = self.calculate_attribute_entropy(attribute, type_filter)

        if not self.start_entropies:
            # Initialize start_entropies if not done already
            self.start_entropies = current_entropies
            return 0

        # Calculate emergence as the difference between start on server.txt and current entropies
        emergence_values = [
            self.start_entropies[agent] - current_entropy
            for agent, current_entropy in current_entropies.items()
            if agent in self.start_entropies
        ]
        return np.mean(emergence_values) if emergence_values else 0


    def calculate_attribute_entropy(self, attribute, type_filter):
        """Compute entropy for each agent for a specific attribute, optionally filtered by agent type."""
        # Filter agents based on the type_filter parameter

        agents = [
            agent for agent in self.schedule.agents
            if isinstance(agent, type_filter)
        ]
        # Return a flat dictionary: {agent: entropy_value}
        return {agent: agent.entropy(attribute) for agent in agents}


    def average_entropy(self, attribute, type_filter):
        """Compute average entropy for a specific attribute, optionally filtered by agent type."""
        # Filter agents based on the type_filter parameter
        agents = [
            agent for agent in self.schedule.agents
            if isinstance(agent, type_filter)
        ]
        # Calculate the entropy
        entropies = [agent.entropy(attribute) for agent in agents]
        return np.mean(entropies) if entropies else 0

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()