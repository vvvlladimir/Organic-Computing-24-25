from mesa import Agent
import numpy as np
import random
import model

class ObjectAgent(Agent):
    def __init__(self, model, object_type):
        super().__init__(model)
        self.object_type = object_type


class AntAgent(Agent):
    def __init__(self, model, step_size=1):
        super().__init__(model)
        self.carrying = None
        self.step_size = step_size

    def neighborhood_function(self):
        """Modified neighborhood function f* as per the requirements in the image."""
        alpha = self.model.ALPHA
        sigma_squared = self.model.SIGMA_SQUARED
        radius1 = int((np.sqrt(sigma_squared) - 1) / 2)
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=radius1)

        # Calculate the modified similarity measure for each neighbor
        similarities = []
        for n in neighbors:
            if isinstance(n, ObjectAgent):
                if self.carrying:
                    obj_type = self.carrying.object_type
                    similarity = 1 - (self.distance(obj_type, n.object_type) / alpha)
                else:
                    # Check the type of ObjectAgent at the ant's position
                    cell_contents = self.model.grid.get_cell_list_contents([self.pos])
                    object_at_pos = [obj for obj in cell_contents if isinstance(obj, ObjectAgent)]

                    if object_at_pos:  # Ensure there is an object at the position
                        obj_type = object_at_pos[0].object_type
                        similarity = 1 - (self.distance(obj_type, n.object_type) / alpha)
                    else:
                        similarity = 1  # Default similarity if no object is under the ant
                similarities.append(similarity)

        # Return the modified similarity function value
        if all(similarity > 0 for similarity in similarities):
            return (1 / sigma_squared) * sum(similarities)
        else:
            return 0.0

    def distance(self, obj1, obj2):
        """Distance (dissimilarity) function between objects"""
        return 0 if obj1 == obj2 else 1

    def pick_up(self):
        """A function for picking up an object by an agent."""
        neighborhood_similarity = self.neighborhood_function()
        k_plus = self.model.PICKUP_THRESHOLD  # соответствует k^+ из формулы
        p_pick = (k_plus / (k_plus + neighborhood_similarity)) ** 2
        return np.random.rand() < p_pick


    def drop(self):
        """Function for dropping an object by an agent."""
        neighborhood_similarity = self.neighborhood_function()
        k_minus = self.model.DROP_THRESHOLD  # соответствует k^- из формулы
        p_drop = (neighborhood_similarity / (k_minus + neighborhood_similarity)) ** 2
        return np.random.rand()  < p_drop

    def move(self, add=0):
        """Move the agent by step_size in a random direction."""
        new_position = (self.pos[0] + random.randint(-self.step_size - add, self.step_size + add),
                        self.pos[1] + random.randint(-self.step_size - add, self.step_size + add))
        self.model.grid.move_agent(self, new_position)

    def step(self):
        # Agent logic per simulation step
        if self.carrying:
            if self.drop():
                self.model.grid.place_agent(self.carrying, self.pos)
                self.carrying = None
            self.move()
        else:
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=1)
            objects = [obj for obj in neighbors if isinstance(obj, ObjectAgent)]
            if objects and self.pick_up():
                self.carrying = random.choice(objects)
                self.model.grid.remove_agent(self.carrying)
            self.move(add=0)
