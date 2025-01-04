from matplotlib.figure import Figure
from mesa.visualization.utils import update_counter
from model import ClusteringModel
from mesa.visualization import SolaraViz, make_space_component
from agents import AntAgent
import solara

# Model parameters
GRID_SIZE = 50
NUM_AGENTS = 100
NUM_OBJECTS = 250

def agent_portrayal(agent):
    """Define the appearance of agents for visualization."""
    colors = {0: "green", 1: "red", 2: "purple"}  # Assign distinct colors to each type
    size = 20

    if isinstance(agent, AntAgent):
        color = colors.get(agent.carrying.object_type, "gray") if agent.carrying else "none"
        return {"size": size * 1.5 if agent.carrying else size, "color": color}
    return {"size": size, "color": colors.get(agent.object_type, "gray")}

@solara.component
def AntEmergenceGraph(model):
    """Line graph visualization for Ant Emergence."""
    update_counter.get()

    fig = Figure()
    ax = fig.subplots()

    df = model.datacollector.get_model_vars_dataframe()

    if not df.empty:
        steps = df.index
        ax.plot(steps, df["Ant_Emergence_X"], label="Ant Emergence (X)", color="blue", linewidth=1.5)
        ax.plot(steps, df["Ant_Emergence_Y"], label="Ant Emergence (Y)", color="green", linewidth=1.5)
        ax.plot(steps, df["Ant_Emergence_Particle"], label="Ant Emergence (Particle)", color="red", linewidth=1.5)

        ax.set_title("Ant Emergence Over Time", fontsize=14)
        ax.set_xlabel("Steps", fontsize=12)
        ax.set_ylabel("Emergence", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No data available yet", ha='center', va='center', fontsize=14)
        ax.axis('off')

    solara.FigureMatplotlib(fig)

@solara.component
def AntObjectEmergenceParticleGraph(model):
    """Line graph visualization for Ant Emergence."""
    update_counter.get()

    fig = Figure()
    ax = fig.subplots()

    df = model.datacollector.get_model_vars_dataframe()

    if not df.empty:
        steps = df.index
        ax.plot(steps, df["OBJ_Average_Entropy_Neighbors"], label="Object Entropy (Neighbors)", color="brown", linewidth=1.5)
        ax.plot(steps, df["Ant_Average_Entropy_Particle"], label="Ant Entropy (Particle)", color="red", linewidth=1.5)

        ax.set_title("Ant & Object Entropy (Particle) Over Time", fontsize=14)
        ax.set_xlabel("Steps", fontsize=12)
        ax.set_ylabel("Emergence", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No data available yet", ha='center', va='center', fontsize=14)
        ax.axis('off')

    solara.FigureMatplotlib(fig)


@solara.component
def AntEntropyGraph(model):
    """Line graph visualization for Ant Average Entropy."""
    update_counter.get()

    fig = Figure()
    ax = fig.subplots()

    df = model.datacollector.get_model_vars_dataframe()

    if not df.empty:
        steps = df.index
        ax.plot(steps, df["Ant_Average_Entropy_X"], label="Ant Entropy (X)", color="blue", linewidth=1.5)
        ax.plot(steps, df["Ant_Average_Entropy_Y"], label="Ant Entropy (Y)", color="green", linewidth=1.5)

        ax.set_title("Ant Average Entropy Over Time", fontsize=14)
        ax.set_xlabel("Steps", fontsize=12)
        ax.set_ylabel("Entropy", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No data available yet", ha='center', va='center', fontsize=14)
        ax.axis('off')

    solara.FigureMatplotlib(fig)


@solara.component
def ObjectEntropyGraph(model):
    """Line graph visualization for Object Average Entropy."""
    update_counter.get()

    fig = Figure()
    ax = fig.subplots()

    df = model.datacollector.get_model_vars_dataframe()

    if not df.empty:
        steps = df.index
        ax.plot(steps, df["OBJ_Average_Entropy_X"], label="Object Entropy (X)", color="orange", linewidth=1.5)
        ax.plot(steps, df["OBJ_Average_Entropy_Y"], label="Object Entropy (Y)", color="purple", linewidth=1.5)

        ax.set_title("Object Average Entropy Over Time", fontsize=14)
        ax.set_xlabel("Steps", fontsize=12)
        ax.set_ylabel("Entropy", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No data available yet", ha='center', va='center', fontsize=14)
        ax.axis('off')

    solara.FigureMatplotlib(fig)


# Model parameters as sliders for user interactivity
model_params = {
    "num_agents": {
        "type": "SliderInt",
        "value": NUM_AGENTS,
        "label": "Number of ants",
        "min": 100,
        "max": 1000,
        "step": 10,
    },
    "width": GRID_SIZE,
    "height": GRID_SIZE,
    "num_objects": {
        "type": "SliderInt",
        "value": NUM_OBJECTS,
        "label": "Number of objects",
        "min": 200,
        "max": 2500,
        "step": 10,
    },
}

# Initialize the model
initial_model = ClusteringModel(
    width=GRID_SIZE,
    height=GRID_SIZE,
    num_agents=NUM_AGENTS,
    num_objects=NUM_OBJECTS
)

# Create a space visualization component
SpaceGraph = make_space_component(agent_portrayal)

# Create the Solara page for visualization with separate graphs
page = SolaraViz(
    initial_model,
    components=[SpaceGraph, AntEmergenceGraph, AntEntropyGraph, ObjectEntropyGraph, AntObjectEmergenceParticleGraph],
    model_params=model_params,
    name="Ant Clustering Visualization with Separate Graphs"
)

page