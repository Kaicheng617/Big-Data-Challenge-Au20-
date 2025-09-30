# Big-Data-Challenge-Au20-

# Au₂₀ Cluster Energy Modeling and Analysis

## Project Overview
This repository contains the implementation for energy modeling and structural analysis of **Au₂₀ (20-atom gold) clusters** as part of the Big Data Challenge. The project applies Graph Neural Networks (GNNs) to predict cluster energies from atomic coordinates and analyzes structural stability.


## Key Components

### Model
- Trained SchNet model checkpoint: `best_schnet_model.pth`
- Performance on test set:
  - RMSE: 1.160369 eV
  - R²: 0.864829
- Model architecture: 6 interaction layers, 128 hidden channels


### Visualizations
All visualizations are stored in the `Au20_Energy_Visualizations/` directory, including:
- Energy distribution visualizations (histograms, KDE plots, box plots, Q-Q plots)
- Structure-specific visualizations:
  - `lowest_energy_visualization.png`: Most stable structure (ID 350, E = -1557.209460 eV)
  - `median_energy_visualization.png`: Median energy structure (ID 464, E = -1551.656057 eV)
  - `highest_energy_visualization.png`: Highest energy structure (ID 78, E = -1530.908363 eV)


### Code
- `Big Data Challenge.py`: Main Python script for:
  - Data processing
  - Graph construction
  - Model training
  - Structural analysis
  - Visualization generation
- `Big Data Challenge.html`: Jupyter Notebook export containing:
  - Exploratory data analysis
  - Model training results
  - Structural stability analysis
  - Perturbation sensitivity analysis


## How to Run

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric
- Matplotlib, Seaborn, NumPy, Pandas

### Execution
Run the main script via the command line:
```bash
python "Big Data Challenge.py"
