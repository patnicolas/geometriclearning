import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser

import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser

# 1. Create a 6-point dataset (forming a rough hexagon)
# This structure guarantees we will see 6 components merge into 1, and 1 clear loop form.
np.random.seed(10)
theta = np.linspace(0, 2 * np.pi, 6, endpoint=False)
r = 1.0 + np.random.normal(0, 0.1, 6)  # Add minor noise
x = r * np.cos(theta)
y = r * np.sin(theta)
data = np.c_[x, y]

# 2. Compute the Vietoris-Rips filtration
result = ripser(data, maxdim=1)
diagrams = result['dgms']


# 3. Custom function to plot the persistence barcode
def plot_persistence_barcode(diagrams, labels=["H0", "H1"]):
    fig, ax = plt.subplots(figsize=(9, 5))

    # Find a reasonable max value to replace 'infinity' for display purposes
    all_finite_deaths = [pt[1] for dgm in diagrams for pt in dgm if pt[1] != np.inf]
    max_finite_death = max(all_finite_deaths) if all_finite_deaths else 1.0
    infinity_val = max_finite_death * 1.2  # Extend slightly past max finite death

    current_y = 0
    colors = ['#1f77b4', '#ff7f0e']  # Blue for H0, Orange for H1
    y_ticks = []
    y_labels = []

    # Iterate through dimensions (H0 and H1)
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue

        dim_start_y = current_y

        # Sort features by lifetime (longest lasting first) for a cleaner look
        sorted_dgm = sorted(dgm, key=lambda pt: pt[1] - pt[0], reverse=True)

        for birth, death in sorted_dgm:
            is_inf = (death == np.inf)
            plot_death = infinity_val if is_inf else death

            # Draw the horizontal bar
            ax.hlines(current_y, birth, plot_death, colors=colors[dim], linewidth=3)

            # If it goes to infinity, draw an arrowhead at the end
            if is_inf:
                ax.annotate('', xy=(plot_death, current_y), xytext=(plot_death - 0.05, current_y),
                            arrowprops=dict(arrowstyle="->", color=colors[dim], lw=3))

            current_y += 1

        # Record middle position of this dimension block for labeling
        y_ticks.append(dim_start_y + (len(dgm) - 1) / 2)
        y_labels.append(labels[dim])

        # Add a subtle visual divider between H0 and H1
        if dim < len(diagrams) - 1:
            ax.axhline(current_y - 0.5, color='gray', linestyle=':', alpha=0.5)

    # Styling the plot
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold')
    ax.set_xlabel("Filtration Value ($\epsilon$)", fontsize=12)
    ax.set_title("Persistence Barcode (6-Point Filtration)", fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Set x-limits safely
    ax.set_xlim(-0.05, infinity_val + 0.1)

    plt.tight_layout()
    plt.show()


# Run the plotting function
plot_persistence_barcode(diagrams)



"""
# Generate toy data & compute diagrams
data = np.random.random((100, 2))
diagrams = ripser(data)['dgms']


def plot_barcode(diagrams):
    fig, ax = plt.subplots(figsize=(10, 5))
    current_y = 0
    colors = ['r', 'b', 'g']  # H0, H1, H2...

    # Loop backwards through dimensions so H0 is at the top or bottom as preferred
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue

        # Optional: handle infinite death values gracefully
        dgm_clean = np.copy(dgm)
        max_finite = np.max(dgm_clean[dgm_clean[:, 1] != np.inf, 1]) if np.any(dgm_clean[:, 1] != np.inf) else 1.0
        dgm_clean[dgm_clean[:, 1] == np.inf, 1] = max_finite * 1.2

        # Plot each bar
        for birth, death in dgm_clean:
            ax.hlines(current_y, birth, death, colors=colors[dim % len(colors)], linewidth=2)
            current_y += 1

        # Add a visual separator between dimensions
        ax.axhline(current_y, color='gray', linestyle='--', alpha=0.3)
        current_y += 1

    ax.set_xlabel("Filtration Value")
    ax.set_ylabel("Features")
    ax.get_yaxis().set_visible(False)  # Hide arbitrary Y-axis index
    plt.title("Persistence Barcode")
    plt.show()


plot_barcode(diagrams)
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser

# 1. Create a 6-point dataset (forming a rough hexagon)
# This structure guarantees we will see 6 components merge into 1, and 1 clear loop form.
np.random.seed(10)
theta = np.linspace(0, 2 * np.pi, 6, endpoint=False)
r = 1.0 + np.random.normal(0, 0.1, 6)  # Add minor noise
x = r * np.cos(theta)
y = r * np.sin(theta)
data = np.c_[x, y]

# 2. Compute the Vietoris-Rips filtration
result = ripser(data, maxdim=1)
diagrams = result['dgms']

# 3. Custom function to plot the persistence barcode
def plot_persistence_barcode(diagrams, labels=["H0", "H1"]):
    fig, ax = plt.subplots(figsize=(9, 5))

    # Find a reasonable max value to replace 'infinity' for display purposes
    all_finite_deaths = [pt[1] for dgm in diagrams for pt in dgm if pt[1] != np.inf]
    max_finite_death = max(all_finite_deaths) if all_finite_deaths else 1.0
    infinity_val = max_finite_death * 1.2  # Extend slightly past max finite death

    current_y = 0
    colors = ['#1f77b4', '#ff7f0e']  # Blue for H0, Orange for H1
    y_ticks = []
    y_labels = []

    # Iterate through dimensions (H0 and H1)
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue

        dim_start_y = current_y

        # Sort features by lifetime (longest lasting first) for a cleaner look
        sorted_dgm = sorted(dgm, key=lambda pt: pt[1] - pt[0], reverse=True)

        for birth, death in sorted_dgm:
            is_inf = (death == np.inf)
            plot_death = infinity_val if is_inf else death

            # Draw the horizontal bar
            ax.hlines(current_y, birth, plot_death, colors=colors[dim], linewidth=3)

            # If it goes to infinity, draw an arrowhead at the end
            if is_inf:
                ax.annotate('', xy=(plot_death, current_y), xytext=(plot_death - 0.05, current_y),
                            arrowprops=dict(arrowstyle="->", color=colors[dim], lw=3))

            current_y += 1

        # Record middle position of this dimension block for labeling
        y_ticks.append(dim_start_y + (len(dgm) - 1) / 2)
        y_labels.append(labels[dim])

        # Add a subtle visual divider between H0 and H1
        if dim < len(diagrams) - 1:
            ax.axhline(current_y - 0.5, color='gray', linestyle=':', alpha=0.5)

    # Styling the plot
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold')
    ax.set_xlabel("Filtration Value ($\epsilon$)", fontsize=12)
    ax.set_title("Persistence Barcode (6-Point Filtration)", fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Set x-limits safely
    ax.set_xlim(-0.05, infinity_val + 0.1)

    plt.tight_layout()
    plt.show()

# Run the plotting function
plot_persistence_barcode(diagrams)
"""