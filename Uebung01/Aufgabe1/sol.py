import solara
from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from mesa.visualization.utils import update_counter
from matplotlib.figure import Figure

from agents import ParticleAgent, AntAgent
from model import AntClusteringModel


def agent_portrayal(agent):
    """Define appearance of agents for visualization."""
    if isinstance(agent, ParticleAgent):
        return {"size": 10, "color": "green", "shape": "circle", "tooltip": "Particle"}  # Green particles
    elif isinstance(agent, AntAgent):
        color = "blue" if agent.carrying else "orange"  # Blue for carrying, orange otherwise
        return {
            "size": 50,
            "color": color,
            "shape": "triangle-up",
            "tooltip": f"Ant (Carrying: {agent.carrying})",
        }


@solara.component
def LineGraphWithAverage(model):
    """Dynamic line graph visualization for ant counts over time with averages."""
    update_counter.get()  # Trigger updates for reactive components

    # Create a Matplotlib figure
    fig = Figure()  # Adjust figsize to make the graph longer
    ax = fig.subplots()

    # Retrieve data from the model's DataCollector
    df = model.datacollector.get_model_vars_dataframe()
    if not df.empty:
        # Extract data for plotting
        steps = df.index  # Time steps
        carrying_ants = df["Carrying Ants"]

        # Calculate running average of carrying ants
        carrying_avg = carrying_ants.expanding().mean()

        # Plot the data
        ax.plot(steps, carrying_ants, label="Carrying Ants", color="blue", linestyle="-")
        ax.plot(steps, carrying_avg, label="Avg Carrying Ants", color="green", linestyle=":")

        # Add title, labels, legend, and grid
        ax.set_title("Ant States Over Time")
        ax.set_ylabel("Count")
        ax.set_xlabel("Steps")
        ax.legend(loc="upper right")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    else:
        # Handle the case when no data is available
        ax.text(0.5, 0.5, "No data available yet", ha='center', va='center', fontsize=12)
        ax.axis('off')

    # Render the figure
    solara.FigureMatplotlib(fig)

# Model parameters for user adjustment
model_params = {
    "num_agents": {
        "type": "SliderInt",  # Ensure type matches Solara's supported components
        "value": 50,          # Set an initial value
        "label": "Number of ants",
        "min": 10,
        "max": 200,
        "step": 1,
    },
    "particle_density": {
        "type": "SliderFloat",
        "value": 0.1,
        "label": "Particle density",
        "min": 0.01,
        "max": 1,
        "step": 0.01,
    },
    "step_size": {
        "type": "SliderInt",
        "value": 1,
        "label": "Step size",
        "min": 1,
        "max": 5,
        "step": 1,
    },
    "jump_distance": {
        "type": "SliderInt",
        "value": 2,
        "label": "Jump distance",
        "min": 2,
        "max": 10,
        "step": 1,
    },
    "central_init": {
        "type": "Checkbox",
        "label": "Central on start on server.txt",
    }
}

# Initialize model and visualization components
initial_model = AntClusteringModel(
    num_agents=50, particle_density=0.1, step_size=1, jump_distance=5, central_init=True
)

# Define the space component
SpaceGraph = make_space_component(agent_portrayal)

# Create the dashboard
page = SolaraViz(
    initial_model,
    components=[SpaceGraph, LineGraphWithAverage],
    model_params=model_params,
    name="Enhanced Ant Clustering Visualization"
)

# Run this script to start on server.txt the Solara dashboard server:
page