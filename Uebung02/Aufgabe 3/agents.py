from mesa import Agent
import numpy as np
import random

# Constants
PICKUP_THRESHOLD = 0.1
DROP_THRESHOLD = 0.3
ALPHA = 0.5
SIGMA_SQUARED = 25
RADIUS = int((np.sqrt(SIGMA_SQUARED) - 1) / 2)


def _neighbor_entropy(neighbors):
    """Shannon entropy for neighbor diversity."""
    object_types = [n.object_type for n in neighbors if isinstance(n, ObjectAgent)]
    if not object_types:
        return 0
    value_counts = np.unique(object_types, return_counts=True)[1]
    probabilities = value_counts / len(object_types)
    return -np.sum(probabilities * np.log2(probabilities))

class ObjectAgent(Agent):
    def __init__(self, model, object_type):
        super().__init__(model)
        self.object_type = object_type

    def entropy(self, attribute):
        """Calculate entropy for the given attribute."""
        if self.pos is None:  # Handle the case where the agent is not on the grid
            return 0  # Or some default value
        p_x = 1 / (self.pos[0] + 0.01)
        p_y = 1 / (self.pos[1] + 0.01)
        if attribute == 'x_position':
            return -p_x * np.log2(p_x)
        elif attribute == 'y_position':
            return -p_y * np.log2(p_y)
        elif attribute == 'particle_carried':
            return 1 if self.carrying else 0
        elif attribute == 'neighbors':
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=RADIUS)
            return _neighbor_entropy(neighbors)


class AntAgent(Agent):
    def __init__(self, model, step_size=1):
        super().__init__(model)
        self.carrying = None
        self.step_size = step_size

    def _neighborhood_function(self, neighbors):
        """Modified neighborhood function f* as per the requirements in the image."""
        # Calculate the modified similarity measure for each neighbor
        similarities = []
        for n in neighbors:
            if isinstance(n, ObjectAgent):
                if self.carrying:
                    obj_type = self.carrying.object_type
                    similarity = 1 - (self._distance(obj_type, n.object_type) / ALPHA)
                else:
                    similarity = 1  # Default similarity if no object is under the ant
                similarities.append(similarity)

        # Return the modified similarity function value
        if all(similarity > 0 for similarity in similarities):
            return (1 / SIGMA_SQUARED) * sum(similarities)
        else:
            return 0.0

    @staticmethod
    def _distance(obj1, obj2):
        """Dissimilarity between two object types."""
        return 0 if obj1 == obj2 else 1

    def entropy(self, attribute):
        """Calculate entropy for the given attribute."""
        p_x = 1 / (self.pos[0] + 0.01)
        p_y = 1 / (self.pos[1] + 0.01)
        if attribute == 'x_position':
            return -p_x * np.log2(p_x)
        elif attribute == 'y_position':
            return -p_y * np.log2(p_y)
        elif attribute == 'particle_carried':
            return 1 if self.carrying else 0
        elif attribute == 'neighbors':
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=RADIUS)
            return _neighbor_entropy(neighbors)


    def _move(self):
        """Move to the position with the lowest entropy."""
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=RADIUS)
        current_entropy = self.entropy("neighbors")
        best_position = None
        lowest_entropy = current_entropy

        for pos in neighbors:
            if self.model.grid.is_cell_empty(pos):
                self.model.grid.move_agent(self, pos)
                entropy = self.entropy("neighbors")
                self.model.grid.move_agent(self, self.pos)
                if entropy < lowest_entropy:
                    best_position, lowest_entropy = pos, entropy

        if best_position:
            self.model.grid.move_agent(self, best_position)
        else:
            self.model.grid.move_agent(self, self.model._random_empty_cell())

    def step(self):
        """Ant's behavior at each step."""
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=RADIUS)

        if self.carrying:
            if self._should_drop(neighbors):
                self.model.grid.place_agent(self.carrying, self.pos)
                self.carrying = None
            self._move()
        else:
            objects = [n for n in neighbors if isinstance(n, ObjectAgent)]
            if objects and self._should_pick_up(neighbors):
                self.carrying = random.choice(objects)
                self.model.grid.remove_agent(self.carrying)
            self._move()

    def _should_pick_up(self, neighbors):
        similarity = self._neighborhood_function(neighbors)
        return np.random.rand() < (PICKUP_THRESHOLD / (PICKUP_THRESHOLD + similarity)) ** 2

    def _should_drop(self, neighbors):
        similarity = self._neighborhood_function(neighbors)
        return np.random.rand() < (similarity / (DROP_THRESHOLD + similarity)) ** 2