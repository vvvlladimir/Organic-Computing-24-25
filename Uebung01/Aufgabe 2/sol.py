from matplotlib.figure import Figure
from mesa.visualization.utils import update_counter

from model import ClusteringModel
from mesa.visualization import SolaraViz, make_space_component
import solara

from agents import AntAgent
import agents

# Model parameters
GRID_SIZE = 100
NUM_AGENTS = 2000
NUM_OBJECTS = 5000



def agent_portrayal(agent):
    """Define appearance of agents for visualization."""
    colors = {0: "green", 1: "red", 2: "purple"}  # Assign distinct colors to each type
    size = 20
    if isinstance(agent, AntAgent):
        if agent.carrying:
            color = colors.get(agent.carrying.object_type, "gray")
            size = size * 1.5
        else:
            color = "none"
        return {
            "size": size,
            "color": color,
        }
    else:
        return {
            "size": size,
            "color": colors.get(agent.object_type, "gray"),
        }


model_params = {
    "num_agents": {
        "type": "SliderInt",  # Ensure type matches Solara's supported components
        "value": NUM_AGENTS,          # Set an initial value
        "label": "Number of ants",
        "min": 10,
        "max": 200,
        "step": 1,
    },
    "width": GRID_SIZE,
    "height": GRID_SIZE,
    "num_objects": {
        "type": "SliderInt",  # Ensure type matches Solara's supported components
        "value": NUM_OBJECTS,          # Set an initial value
        "label": "Number of objects",
        "min": 10,
        "max": 200,
        "step": 1,
    }

}
initial_model = ClusteringModel(
    width=GRID_SIZE,
    height=GRID_SIZE,
    num_agents=NUM_AGENTS,
    num_objects=NUM_OBJECTS
)
SpaceGraph = make_space_component(agent_portrayal)

page = SolaraViz(
    initial_model,
    components=[SpaceGraph],
    model_params=model_params,
    name="Enhanced Ant Clustering Visualization"
)
page