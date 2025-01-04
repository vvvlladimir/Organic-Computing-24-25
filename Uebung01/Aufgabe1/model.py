import random
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from agents import ParticleAgent, AntAgent

def count_particles(model):
    particles = sum(isinstance(agent, ParticleAgent) for agent in model.schedule.agents)
    carrying_ants = sum(isinstance(agent, AntAgent) for agent in model.schedule.agents if agent.carrying)
    idle_ants = sum(isinstance(agent, AntAgent) for agent in model.schedule.agents if not agent.carrying)
    return {"Particles": particles, "Carrying Ants": carrying_ants, "Idle Ants": idle_ants}

class AntClusteringModel(Model):
    """Ant Clustering Model with Data Collection for Visualization"""
    def __init__(self, num_agents=50, particle_density=0.1, step_size=1, jump_distance=5, central_init=False):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(50, 50, torus=True)
        self.schedule = SimultaneousActivation(self)

        # Populate grid with particles
        for _, coords in self.grid.coord_iter():
            if random.random() < particle_density:
                particle = ParticleAgent(self)
                self.grid.place_agent(particle, coords)

        # Add ant agents to the grid
        for i in range(self.num_agents):
            ant = AntAgent(self, step_size=step_size, jump_distance=jump_distance)
            if central_init:
                self.grid.place_agent(ant, (25, 25))
            else:
                self.grid.place_agent(ant, (random.randint(0, 49), random.randint(0, 49)))
            self.schedule.add(ant)

        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={"Particles": lambda m: count_particles(m)["Particles"],
                             "Carrying Ants": lambda m: count_particles(m)["Carrying Ants"],
                             "Idle Ants": lambda m: count_particles(m)["Idle Ants"]},
        )

    def step(self):
        """Advance the model by one step and collect data."""
        self.datacollector.collect(self)
        self.schedule.step()
