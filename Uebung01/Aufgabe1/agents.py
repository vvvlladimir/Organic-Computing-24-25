import random
from mesa import Agent


class ParticleAgent(Agent):
    """An agent representing a particle"""
    def __init__(self, model):
        super().__init__(model)


class AntAgent(Agent):
    """An agent representing an ant"""
    def __init__(self, model, step_size=1, jump_distance=5):
        super().__init__(model)
        self.carrying = None
        self.step_size = step_size
        self.jump_distance = jump_distance

    def step(self):
        # If the ant is not carrying a load and is on a cage with a particle
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        particles = [obj for obj in cellmates if isinstance(obj, ParticleAgent)]
        if not self.carrying and particles:
            self.carrying = particles[0]  # Take the particle
            self.model.grid.remove_agent(self.carrying)
            self.jump()
        elif self.carrying:
            # If ant are carrying a load and find an empty seat
            empty_neighbors = [pos for pos in self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
                               if self.model.grid.is_cell_empty(pos)]
            if empty_neighbors:
                new_position = random.choice(empty_neighbors)
                self.model.grid.place_agent(self.carrying, new_position)
                self.carrying = None  # Drop the load
                self.jump()
        else:
            # Move by step_size in a random direction
            self.move()
    def jump(self):
        new_position = (self.pos[0] + random.randint(-self.jump_distance, self.jump_distance),
                        self.pos[1] + random.randint(-self.jump_distance, self.jump_distance))
        self.model.grid.move_agent(self, new_position)

    def move(self):
        """ Step on step_size in a random direction. """
        new_position = (self.pos[0] + random.randint(-self.step_size, self.step_size),
                        self.pos[1] + random.randint(-self.step_size, self.step_size))
        self.model.grid.move_agent(self, new_position)