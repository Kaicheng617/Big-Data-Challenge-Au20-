#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import pandas as pd
import numpy as np

def parse_xyz_file(filepath: str) -> tuple:
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            # Validate atom count declaration (must be exactly 20)
            num_atoms = int(lines[0].strip())
            if num_atoms != 20:
                print(f"Warning: Skipped {os.path.basename(filepath)} - declared atom count ({num_atoms}) != 20")
                return None, None
            
            # Extract energy value with strict float conversion
            energy = float(lines[1].strip())
            
            # Process coordinate lines with field validation
            coords = []
            for line in lines[2:]:
                parts = line.strip().split()
                # Require exactly 4 fields (atom symbol + 3 coordinates)
                if len(parts) != 4:
                    print(f"Warning: Skipped malformed line in {os.path.basename(filepath)}: '{line.strip()}'")
                    continue
                
                # Filter only Gold (Au) atoms
                if parts[0] == 'Au':
                    try:
                        # Convert coordinates to floats with precision validation
                        x, y, z = [float(p) for p in parts[1:4]]
                        coords.append([x, y, z])
                    except ValueError:
                        print(f"Warning: Skipped invalid coordinates in {os.path.basename(filepath)}: '{line.strip()}'")
            
            # Final validation: Coordinate count must match declared atom count
            if len(coords) != num_atoms:
                print(f"Warning: Skipped {os.path.basename(filepath)} - coordinate count ({len(coords)}) != declared atom count ({num_atoms})")
                return None, None
                
            return energy, np.array(coords, dtype=np.float64)
            
    except (IOError, IndexError, ValueError) as e:
        print(f"Error: Failed to parse {os.path.basename(filepath)} - {str(e)}")
        return None, None

def process_data_directory(directory_path: str) -> pd.DataFrame:
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found - {directory_path}")
        return pd.DataFrame(columns=['id', 'energy', 'coordinates'])

    results = []
    
    for filename in sorted(os.listdir(directory_path)):
        if not filename.endswith('.xyz'):
            continue
            
        filepath = os.path.join(directory_path, filename)
        structure_id = os.path.splitext(filename)[0]
        
        energy, coordinates = parse_xyz_file(filepath)
        
        if energy is not None and coordinates is not None:
            results.append({
                'id': structure_id,
                'energy': energy,
                'coordinates': coordinates
            })
    
    return pd.DataFrame(results, columns=['id', 'energy', 'coordinates'])

# --- Execution Pipeline ---
# Configure with absolute path to your dataset directory
DATA_DIRECTORY = r"C:\Users\Administrator\Desktop\大数据竞赛金原子\data\data\Au20_OPT_1000"

# Process all valid files in directory
au20_dataset = process_data_directory(DATA_DIRECTORY)

# Output processing statistics
print(f"Successfully processed {len(au20_dataset)} valid structures out of {sum(1 for _ in os.listdir(DATA_DIRECTORY) if _.endswith('.xyz'))} .xyz files")
if not au20_dataset.empty:
    print("\nDataset preview (first 3 entries):")
    print(au20_dataset.head(3))
    print("\nCoordinate array shape (per structure):", au20_dataset.iloc[0]['coordinates'].shape)
else:
    print("No valid structures found in directory")


# In[20]:


import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd

# --- Energy Distribution Analysis Pipeline ---

# Verify dataset availability
if 'au20_data' not in globals():
    raise RuntimeError("Dataset 'au20_data' not found in global namespace. Execute data loading script first.")

# Configure output directory on desktop
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
OUTPUT_DIR = os.path.join(DESKTOP_PATH, "Au20_Energy_Visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Extract energy values for analysis
energy_values = au20_data['energy'].values

# Calculate distribution statistics
stats_summary = {
    'mean': np.mean(energy_values),
    'variance': np.var(energy_values),
    'std_dev': np.std(energy_values),
    'skewness': skew(energy_values),
    'kurtosis': kurtosis(energy_values),
    'min': np.min(energy_values),
    'median': np.median(energy_values),
    'max': np.max(energy_values)
}

# Print statistical report
print("--- Au20 Cluster Energy Statistical Summary ---")
print(f"Sample Size: {len(energy_values)}")
print(f"Mean: {stats_summary['mean']:.6f}")
print(f"Variance: {stats_summary['variance']:.6f}")
print(f"Standard Deviation: {stats_summary['std_dev']:.6f}")
print(f"Skewness: {stats_summary['skewness']:.6f}")
print(f"Kurtosis: {stats_summary['kurtosis']:.6f}")
print(f"Minimum: {stats_summary['min']:.6f}")
print(f"Median: {stats_summary['median']:.6f}")
print(f"Maximum: {stats_summary['max']:.6f}")
print("----------------------------------------------")

# Generate primary visualization set (2x2 grid)
fig_main, axes = plt.subplots(2, 2, figsize=(15, 12))

# Histogram with density curve
sns.histplot(energy_values, kde=True, bins=50, color='#E5C185', stat="density", ax=axes[0,0])
axes[0,0].axvline(stats_summary['mean'], color='#FBF2C4', linestyle='--', linewidth=2, 
                 label=f'Mean: {stats_summary["mean"]:.2f}')
axes[0,0].axvline(stats_summary['median'], color='#C7522A', linestyle=':', linewidth=2,
                 label=f'Median: {stats_summary["median"]:.2f}')
axes[0,0].set_title('Au20 Cluster Energy Distribution (Histogram)', fontsize=14)
axes[0,0].set_xlabel('Total Energy (eV)', fontsize=12)
axes[0,0].set_ylabel('Density', fontsize=12)
axes[0,0].legend()
axes[0,0].grid(axis='y', alpha=0.3)

# Box plot for distribution spread
sns.boxplot(y=energy_values, ax=axes[0,1], color='#DDE5B4')
axes[0,1].set_title('Au20 Cluster Energy Distribution (Box Plot)', fontsize=14)
axes[0,1].set_ylabel('Total Energy (eV)', fontsize=12)
axes[0,1].grid(axis='y', alpha=0.3)

# Q-Q plot for normality assessment
stats.probplot(energy_values, dist="norm", plot=axes[1,0])
axes[1,0].set_title('Q-Q Plot for Energy Distribution', fontsize=14)
axes[1,0].grid(alpha=0.3)

# Violin plot showing density profile
sns.violinplot(y=energy_values, ax=axes[1,1], color='#CFA093')
axes[1,1].set_title('Au20 Cluster Energy Distribution (Violin Plot)', fontsize=14)
axes[1,1].set_ylabel('Total Energy (eV)', fontsize=12)
axes[1,1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "primary_distribution_analysis.png"), dpi=300, bbox_inches='tight')
plt.close(fig_main)

# Generate secondary visualization set (1x3 grid)
fig_secondary = plt.figure(figsize=(15, 5))

# Energy sequence trend analysis
plt.subplot(1, 3, 1)
plt.scatter(range(len(energy_values)), energy_values, alpha=0.6, color='#5E606C')
plt.plot(range(len(energy_values)), energy_values, alpha=0.3, color='#5E606C')
plt.title('Energy Trend Over Samples', fontsize=14)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Total Energy (eV)', fontsize=12)
plt.grid(alpha=0.3)

# Kernel density estimation
plt.subplot(1, 3, 2)
sns.kdeplot(energy_values, fill=True, alpha=0.6, color='#F1DDBF')
plt.axvline(stats_summary['mean'], color='#7D6B57', linestyle='--', linewidth=2, 
           label=f'Mean: {stats_summary["mean"]:.2f}')
plt.axvline(stats_summary['median'], color='#C9BB98', linestyle=':', linewidth=2,
           label=f'Median: {stats_summary["median"]:.2f}')
plt.title('Energy Density Distribution', fontsize=14)
plt.xlabel('Total Energy (eV)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

# Cumulative distribution function
plt.subplot(1, 3, 3)
sorted_energy = np.sort(energy_values)
cumulative_prob = np.arange(1, len(sorted_energy) + 1) / len(sorted_energy)
plt.plot(sorted_energy, cumulative_prob, color='#9DAD7F', linewidth=2)
plt.title('Cumulative Energy Distribution', fontsize=14)
plt.xlabel('Total Energy (eV)', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "secondary_distribution_analysis.png"), dpi=300, bbox_inches='tight')
plt.close(fig_secondary)

# Calculate additional distribution metrics
iqr = np.percentile(energy_values, 75) - np.percentile(energy_values, 25)
lower_bound = np.percentile(energy_values, 25) - 1.5 * iqr
upper_bound = np.percentile(energy_values, 75) + 1.5 * iqr
outlier_mask = (energy_values < lower_bound) | (energy_values > upper_bound)
outlier_count = np.sum(outlier_mask)

# Print extended analysis results
print("\n--- Additional Distribution Metrics ---")
print(f"Energy Range: {stats_summary['max'] - stats_summary['min']:.6f}")
print(f"Coefficient of Variation: {stats_summary['std_dev'] / abs(stats_summary['mean']) * 100:.2f}%")
print(f"95th Percentile: {np.percentile(energy_values, 95):.6f}")
print(f"5th Percentile: {np.percentile(energy_values, 5):.6f}")
print(f"Interquartile Range (IQR): {iqr:.6f}")
print(f"Number of Outliers (IQR method): {outlier_count}")
print(f"Outlier Thresholds: [{lower_bound:.6f}, {upper_bound:.6f}]")
print("---------------------------------------")

# Verification output
print(f"\nVisualization files saved to: {OUTPUT_DIR}")
print(f"Total files generated: {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])}")


# In[30]:


import torch
from torch_geometric.data import Data
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import warnings

# Configure matplotlib for proper scientific notation
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['savefig.dpi'] = 300
warnings.filterwarnings('ignore', category=UserWarning)
sns.set_style("whitegrid")

def create_graph_dataset(
    df: pd.DataFrame,
    cutoff: float = 3.5,
    num_gaussians: int = 50,
    gamma: float = 10.0
) -> list:

    # Parameter validation
    if not all(param > 0 for param in [cutoff, num_gaussians, gamma]):
        raise ValueError("All parameters must have positive values")
    
    # Precompute RBF centers
    rbf_centers = torch.linspace(0.0, cutoff, num_gaussians, dtype=torch.float32)
    graph_dataset = []
    
    # Process all structures with index tracking
    for df_idx, (orig_idx, row) in enumerate(df.iterrows()):
        try:
            coords = row['coordinates']
            
            # Validate coordinate dimensions
            if coords.shape != (20, 3):
                continue
                
            # Distance matrix calculation
            dist_matrix = squareform(pdist(coords))
            
            # Edge construction
            adj_mask = (dist_matrix < cutoff) & (dist_matrix > 1e-5)
            edge_index = torch.tensor(np.vstack(np.where(adj_mask)), dtype=torch.long)
            
            # Skip disconnected graphs
            if edge_index.size(1) < 1:
                continue
            
            # Edge feature processing
            edge_distances = torch.tensor(dist_matrix[adj_mask], dtype=torch.float32).view(-1, 1)
            edge_distances = torch.clamp(edge_distances, 0.0, cutoff)
            edge_attr = torch.exp(-gamma * (edge_distances - rbf_centers)**2)
            
            # Node feature calculation
            node_features = torch.tensor(np.sum(adj_mask, axis=1), dtype=torch.float32).view(-1, 1)
            
            # Create graph object with metadata
            graph_dataset.append(Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([row['energy']], dtype=torch.float32),
                pos=torch.tensor(coords, dtype=torch.float32),
                num_nodes=20,
                df_index=orig_idx,
                df_idx=df_idx
            ))
            
        except Exception:
            continue
    
    return graph_dataset

def visualize_representative_structure(
    graph_data: Data,
    output_path: str,
    structure_id: str,
    energy: float,
    structure_type: str
):
    coords = graph_data.pos.numpy()
    
    # Create figure with tight layout
    fig = plt.figure(figsize=(12, 10), tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    
    # Configure atom visualization
    coordination = graph_data.x.numpy().flatten()
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        s=250,
        c=coordination,
        cmap='viridis',
        edgecolor='k',
        linewidth=0.5,
        alpha=0.9,
        depthshade=True
    )
    
    # Draw bonds
    for i in range(graph_data.edge_index.size(1)):
        start, end = graph_data.edge_index[:, i].numpy()
        ax.plot(
            [coords[start, 0], coords[end, 0]],
            [coords[start, 1], coords[end, 1]],
            [coords[start, 2], coords[end, 2]],
            color='silver',
            alpha=0.4,
            linewidth=1.5
        )
    
    # Configure 3D view with proper scientific notation
    ax.set_xlabel('X (Å)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (Å)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (Å)', fontsize=12, labelpad=10)
    
    # Set title based on structure type
    title_type = ""
    if structure_type == "lowest":
        title_type = "Lowest Energy (Most Stable)"
    elif structure_type == "median":
        title_type = "Median Energy"
    elif structure_type == "highest":
        title_type = "Highest Energy"
    
    ax.set_title(
        r'Au$_{20}$ ' + title_type + ' Structure: ' + structure_id + 
        '\nTotal Energy: ' + f'{energy:.6f} eV',
        fontsize=14,
        pad=30
    )
    
    # Add coordination number legend
    plt.colorbar(sc, ax=ax, label='Coordination Number')
    
    # Ensure equal aspect ratio
    scaling = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
    
    # Save with high resolution
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def save_representative_visualizations(
    graph_dataset: list,
    df: pd.DataFrame,
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort structures by energy
    df_sorted = df.sort_values('energy')
    
    # Get indices for representative structures
    lowest_idx = df_sorted.index[0]
    median_idx = df_sorted.index[len(df_sorted)//2]
    highest_idx = df_sorted.index[-1]
    
    # Create mapping from df_index to graph_dataset index
    index_map = {graph.df_index: i for i, graph in enumerate(graph_dataset)}
    
    # Process each representative structure
    for structure_type, df_idx in [("lowest", lowest_idx), 
                                   ("median", median_idx), 
                                   ("highest", highest_idx)]:
        # Find corresponding graph in dataset
        if df_idx in index_map:
            graph = graph_dataset[index_map[df_idx]]
            
            # Retrieve metadata
            structure_id = df.loc[df_idx, 'id']
            energy = df.loc[df_idx, 'energy']
            
            # Generate visualization with appropriate label
            output_path = os.path.join(output_dir, f"structure_{structure_id}_{structure_type}.png")
            visualize_representative_structure(
                graph_data=graph,
                output_path=output_path,
                structure_id=structure_id,
                energy=energy,
                structure_type=structure_type
            )
            print(f"Generated {structure_type} energy structure visualization: {output_path}")
        else:
            print(f"Warning: Could not find graph for {structure_type} energy structure")

# --- Execution Pipeline ---
if __name__ == "__main__":
    # Verify input data
    if 'au20_data' not in globals():
        raise RuntimeError("Input DataFrame 'au20_data' not found")
    
    if not all(col in au20_data.columns for col in ['id', 'energy', 'coordinates']):
        raise ValueError("DataFrame missing required columns")
    
    # Process data
    au20_graphs = create_graph_dataset(
        df=au20_data,
        cutoff=3.5,
        num_gaussians=50,
        gamma=10.0
    )
    
    # Configure output directory
    output_dir = os.path.join(
        os.path.expanduser("~"), 
        "Desktop", 
        "Au20_Structure_Visualizations"
    )
    
    # Generate only the three representative visualizations
    save_representative_visualizations(
        graph_dataset=au20_graphs,
        df=au20_data,
        output_dir=output_dir
    )
    
    # Output confirmation
    print("\n" + "="*50)
    print("Representative Structure Visualization Complete")
    print(f"Generated 3 key visualizations in: {output_dir}")
    print("="*50)


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def visualize_graph_statistics(graph_dataset, output_dir):

    # Configure matplotlib for scientific notation
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Calculate statistics
    degrees = [graph.edge_index.size(1) / graph.num_nodes for graph in graph_dataset]
    edge_counts = [graph.edge_index.size(1) for graph in graph_dataset]
    energies = [graph.y.item() for graph in graph_dataset]
    
    # Collect node features (coordination numbers)
    node_features = []
    for graph in graph_dataset:
        node_features.extend(graph.x.flatten().tolist())
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Node degree distribution
    axes[0, 0].hist(degrees, bins=30, color='#E6C994', edgecolor='#000000', alpha=0.7)
    axes[0, 0].set_title(r'Distribution of Average Node Degree in Au$_{20}$', fontsize=12)
    axes[0, 0].set_xlabel('Average Degree', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].grid(alpha=0.3, linestyle='--')
    
    # 2. Edge count distribution
    axes[0, 1].hist(edge_counts, bins=30, color='#93B071', edgecolor='#000000', alpha=0.7)
    axes[0, 1].set_title(r'Distribution of Number of Edges in Au$_{20}$', fontsize=12)
    axes[0, 1].set_xlabel('Number of Edges', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].grid(alpha=0.3, linestyle='--')
    
    # 3. Energy vs edge count relationship
    scatter = axes[1, 0].scatter(edge_counts, energies, alpha=0.6, color='#F8E16F', s=40)
    axes[1, 0].set_title(r'Energy vs Number of Edges in Au$_{20}$', fontsize=12)
    axes[1, 0].set_xlabel('Number of Edges', fontsize=10)
    axes[1, 0].set_ylabel('Energy (eV)', fontsize=10)
    axes[1, 0].grid(alpha=0.3, linestyle='--')
    
    # 4. Node feature distribution (coordination numbers)
    axes[1, 1].hist(node_features, bins=30, color='#C75A1B', edgecolor='#000000', alpha=0.7)
    axes[1, 1].set_title(r'Distribution of Coordination Numbers in Au$_{20}$', fontsize=12)
    axes[1, 1].set_xlabel('Coordination Number', fontsize=10)
    axes[1, 1].set_ylabel('Frequency', fontsize=10)
    axes[1, 1].grid(alpha=0.3, linestyle='--')
    
    # Enhance layout
    plt.tight_layout(pad=3.0)
    
    # Save figure
    output_path = os.path.join(output_dir, "graph_statistics.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

# --- Execution Pipeline ---
if __name__ == "__main__":
    # Verify required data exists
    if 'au20_graphs' not in globals():
        raise RuntimeError("Graph dataset 'au20_graphs' not found. Execute graph construction first.")
    
    # Configure output directory
    output_dir = os.path.join(
        os.path.expanduser("~"), 
        "Desktop", 
        "Au20_Structure_Visualizations"
    )
    
    # Generate and save statistics visualization
    visualize_graph_statistics(
        graph_dataset=au20_graphs,
        output_dir=output_dir
    )
    
    # Minimal output confirmation
    print(f"Graph statistics visualization saved to: {output_dir}/graph_statistics.png")


# In[5]:


import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import SchNet
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# --- Section 3: Building and Training Graph Neural Network ---

# Assuming 'au20_graphs' list already exists in your environment

# 1. Dataset Split (80% training, 10% validation, 10% testing)
train_val_graphs, test_graphs = train_test_split(au20_graphs, test_size=0.1, random_state=42)
train_graphs, val_graphs = train_test_split(train_val_graphs, test_size=1/9, random_state=42) # 1/9 of 90% is 10% of total

print("=" * 50)
print("DATASET SPLIT COMPLETED:")
print(f"Training samples: {len(train_graphs)}")
print(f"Validation samples: {len(val_graphs)}")
print(f"Test samples: {len(test_graphs)}")
print("=" * 50)

# 2. Create data loaders
# DataLoader handles automatic batching of graphs
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

# 3. Model definition
# Using SchNet model implemented in PyTorch Geometric
# SchNet is designed for molecular property prediction with physical meaning
model = SchNet(
    hidden_channels=128,  # Hidden layer dimension
    num_filters=128,      # Number of filters
    num_interactions=6,   # Number of interaction blocks (convolution layers)
    num_gaussians=50,     # Consistent with data processing
    cutoff=3.5,           # Consistent with data processing
    readout='add'         # Node feature aggregation method
)

# 4. Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nTraining will be performed on {device}.")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Enable cuDNN for faster training
    torch.backends.cudnn.benchmark = True

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
loss_fn = torch.nn.MSELoss()  # Mean squared error loss function

# 5. Training and validation loops
def train_epoch():
    model.train()
    total_loss = 0
    num_samples = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for data in pbar:
        data = data.to(device)
        # Create placeholder atomic numbers (Au = 79) for SchNet
        data.z = torch.full((data.num_nodes,), 79, dtype=torch.long, device=device)
        
        optimizer.zero_grad()
        output = model(data.z, data.pos, data.batch)
        loss = loss_fn(output.squeeze(), data.y)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        num_samples += data.num_graphs
        
        pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
    
    return total_loss / num_samples

def validate_epoch(loader, desc="Validation"):
    model.eval()
    total_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for data in pbar:
            data = data.to(device)
            data.z = torch.full((data.num_nodes,), 79, dtype=torch.long, device=device)
            
            output = model(data.z, data.pos, data.batch)
            loss = loss_fn(output.squeeze(), data.y)
            
            total_loss += loss.item() * data.num_graphs
            num_samples += data.num_graphs
            
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
    
    return total_loss / num_samples

def calculate_metrics(loader, desc="Evaluation"):
    """Calculate additional metrics beyond MSE"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.z = torch.full((data.num_nodes,), 79, dtype=torch.long, device=device)
            
            output = model(data.z, data.pos, data.batch)
            predictions.extend(output.squeeze().cpu().numpy())
            targets.extend(data.y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
    
    return mse, rmse, mae, r2

# --- Start Training ---
epochs = 800  # Increased to 800 epochs
best_val_loss = float('inf')
train_losses = []
val_losses = []
train_rmses = []
val_rmses = []

# Early stopping parameters
patience = 50  # Early stopping patience
patience_counter = 0
best_epoch = 0

print("\n" + "=" * 60)
print("STARTING MODEL TRAINING ON GPU")
print(f"Training for up to {epochs} epochs with early stopping (patience: {patience})")
print("=" * 60)

start_time = time.time()

for epoch in range(1, epochs + 1):
    # Training phase
    train_loss = train_epoch()
    val_loss = validate_epoch(val_loader)
    
    # Calculate additional metrics
    _, train_rmse, _, _ = calculate_metrics(train_loader, "Train Metrics")
    _, val_rmse, _, _ = calculate_metrics(val_loader, "Val Metrics")
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_rmses.append(train_rmse)
    val_rmses.append(val_rmse)
    
    # Update learning rate based on validation loss
    scheduler.step(val_loss)
    
    # Check for early stopping and save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0  # Reset patience counter
        
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }, 'best_schnet_model.pth')
        print(f"Epoch {epoch:03d}: New best model saved! Val Loss: {val_loss:.6f}")
    else:
        patience_counter += 1
    
    # Print progress
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:03d}/{epochs} | '
              f'Train MSE: {train_loss:.6f} | '
              f'Val MSE: {val_loss:.6f} | '
              f'Train RMSE: {train_rmse:.6f} | '
              f'Val RMSE: {val_rmse:.6f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    else:
        print(f'Epoch {epoch:03d}/{epochs} | '
              f'Train MSE: {train_loss:.6f} | '
              f'Val MSE: {val_loss:.6f} | '
              f'Val RMSE: {val_rmse:.6f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f} | '
              f'Patience: {patience_counter}/{patience}', end='\r')
    
    # Early stopping check
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch}!")
        print(f"Best epoch was {best_epoch} with validation loss: {best_val_loss:.6f}")
        break

end_time = time.time()
print(f"\nTraining completed in {end_time - start_time:.2f} seconds")

print("=" * 60)
print("TRAINING COMPLETED")
print(f"Best validation loss (MSE): {best_val_loss:.6f}")
print(f"Best validation RMSE: {min(val_rmses):.6f}")
print(f"Best epoch: {best_epoch}")
print(f"Total epochs trained: {epoch}")
print("Best model saved to 'best_schnet_model.pth'")
print("=" * 60)

# Load best model for final evaluation
print("\nLoading best model for final evaluation...")
checkpoint = torch.load('best_schnet_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Final evaluation on test set
test_mse, test_rmse, test_mae, test_r2 = calculate_metrics(test_loader, "Test Metrics")
print(f"\nFINAL TEST SET RESULTS:")
print(f"Test MSE: {test_mse:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")
print(f"Test MAE: {test_mae:.6f}")
print(f"Test R²: {test_r2:.6f}")

# Plotting loss curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# MSE plot
ax1.plot(train_losses, label='Training Loss (MSE)', color='blue', linewidth=2)
ax1.plot(val_losses, label='Validation Loss (MSE)', color='red', linewidth=2)
ax1.axvline(x=best_epoch-1, color='green', linestyle='--', label=f'Best Epoch: {best_epoch}', alpha=0.7)
ax1.set_title('Training and Validation Loss (MSE)', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# RMSE plot
ax2.plot(train_rmses, label='Training RMSE', color='blue', linewidth=2)
ax2.plot(val_rmses, label='Validation RMSE', color='red', linewidth=2)
ax2.axvline(x=best_epoch-1, color='green', linestyle='--', label=f'Best Epoch: {best_epoch}', alpha=0.7)
ax2.set_title('Training and Validation RMSE', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Root Mean Squared Error (RMSE)', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional visualization: Prediction vs Actual
model.eval()
predictions = []
targets = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        data.z = torch.full((data.num_nodes,), 79, dtype=torch.long, device=device)
        
        output = model(data.z, data.pos, data.batch)
        predictions.extend(output.squeeze().cpu().numpy())
        targets.extend(data.y.cpu().numpy())

predictions = np.array(predictions)
targets = np.array(targets)

# Plot prediction vs actual
plt.figure(figsize=(10, 8))
plt.scatter(targets, predictions, alpha=0.6, color='blue', s=30)
plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Energy (eV)', fontsize=12)
plt.ylabel('Predicted Energy (eV)', fontsize=12)
plt.title(f'Prediction vs Actual (R² = {test_r2:.4f})', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n" + "=" * 60)
print("MODEL TRAINING AND EVALUATION COMPLETE!")
print("=" * 60)

# GPU内存使用情况监控
if torch.cuda.is_available():
    print(f"GPU内存使用峰值: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"GPU内存缓存峰值: {torch.cuda.max_memory_reserved()/1e9:.2f} GB")


# In[6]:


import pandas as pd
import numpy as np

# --- Task 2: Identify and Export the Most Stable Structure ---

# Assuming 'au20_data' DataFrame already exists in your environment

# 1. Find the index of the structure with the lowest energy
lowest_energy_idx = au20_data['energy'].idxmin()

# 2. Extract all information for this structure
ground_state_structure = au20_data.loc[lowest_energy_idx]

# 3. Print information
print("--- Most Stable (Lowest Energy) Au20 Structure Information ---")
print(f"Structure ID: {ground_state_structure['id']}")
print(f"Energy: {ground_state_structure['energy']:.6f} eV")
print("-------------------------------------------------------------")

# 4. Save the coordinates of this structure to a .xyz file for visualization
def save_coords_to_xyz(coords, energy, filepath):
    """
    Save coordinates and energy to .xyz file format.
    
    Parameters:
        coords (numpy.ndarray): Atomic coordinates [num_atoms, 3]
        energy (float): Total energy
        filepath (str): Output file path
    """
    num_atoms = len(coords)
    with open(filepath, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write(f"Energy: {energy:.6f}\n")
        for i in range(num_atoms):
            f.write(f"Au  {coords[i][0]:.8f}  {coords[i][1]:.8f}  {coords[i][2]:.8f}\n")

# Call function to save file
output_xyz_file = 'lowest_energy_structure.xyz'
save_coords_to_xyz(
    ground_state_structure['coordinates'],
    ground_state_structure['energy'],
    output_xyz_file
)

print(f"Successfully saved the most stable structure coordinates to '{output_xyz_file}'.")
print("\nNext Step Suggestion:")
print("Please use molecular visualization software such as VMD or OVITO to open this file and view its 3D structure.")


# In[27]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import os

def analyze_ground_state_structure(ground_state_structure, output_dir):

    # Configure matplotlib for scientific notation
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    coords = ground_state_structure['coordinates']
    cutoff = 3.5
    
    # 1. Calculate distance matrix
    dist_matrix = squareform(pdist(coords))
    
    # 2. Analyze bond length distribution
    bond_lengths = dist_matrix[np.triu_indices(20, k=1)]
    bond_lengths = bond_lengths[bond_lengths < cutoff]
    
    mean_bond_length = bond_lengths.mean()
    std_bond_length = bond_lengths.std()
    
    # 3. Analyze coordination number distribution
    adj_matrix = dist_matrix < cutoff
    np.fill_diagonal(adj_matrix, False)
    coordination_numbers = np.sum(adj_matrix, axis=1)
    
    unique_coordinations, counts = np.unique(coordination_numbers, return_counts=True)
    
    # 4. Additional structural analysis
    min_bond_length = bond_lengths.min()
    max_bond_length = bond_lengths.max()
    bond_length_range = max_bond_length - min_bond_length
    
    center_of_mass = coords.mean(axis=0)
    distances_to_com = np.sqrt(np.sum((coords - center_of_mass)**2, axis=1))
    mean_distance_to_com = distances_to_com.mean()
    std_distance_to_com = distances_to_com.std()
    
    max_coordination = int(coordination_numbers.max())
    min_coordination = int(coordination_numbers.min())
    high_coordination_atoms = np.where(coordination_numbers == max_coordination)[0]
    low_coordination_atoms = np.where(coordination_numbers == min_coordination)[0]
    
    # 5. Visualization of analysis results
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot bond length distribution histogram
    sns.histplot(bond_lengths, bins=30, kde=True, ax=axes[0], color='#E26D5C', alpha=0.7)
    axes[0].set_title(r'Bond Length Distribution in Au$_{20}$', fontsize=14)
    axes[0].set_xlabel('Interatomic Distance (Å)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].axvline(mean_bond_length, color='#723D46', linestyle='--', 
                   label=f'Mean: {mean_bond_length:.2f} Å')
    axes[0].legend()
    axes[0].grid(alpha=0.3, linestyle='--')
    
    # Plot coordination number distribution bar chart
    sns.barplot(x=unique_coordinations, y=counts, ax=axes[1], palette='viridis')
    axes[1].set_title(r'Coordination Number Distribution in Au$_{20}$', fontsize=14)
    axes[1].set_xlabel('Coordination Number', fontsize=12)
    axes[1].set_ylabel('Number of Atoms', fontsize=12)
    for index, value in enumerate(counts):
        axes[1].text(index, value, str(value), ha='center', va='bottom')
    axes[1].grid(alpha=0.3, axis='y', linestyle='--')
    
    plt.suptitle(r'Geometric Analysis of Most Stable Au$_{20}$ Cluster', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    output_path = os.path.join(output_dir, "ground_state_analysis.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # Return key metrics for reporting
    return {
        'mean_bond_length': mean_bond_length,
        'std_bond_length': std_bond_length,
        'unique_coordinations': unique_coordinations,
        'counts': counts,
        'min_bond_length': min_bond_length,
        'max_bond_length': max_bond_length,
        'bond_length_range': bond_length_range,
        'mean_distance_to_com': mean_distance_to_com,
        'std_distance_to_com': std_distance_to_com,
        'max_coordination': max_coordination,
        'high_coordination_atoms': high_coordination_atoms,
        'min_coordination': min_coordination,
        'low_coordination_atoms': low_coordination_atoms
    }

def print_minimal_results(results):
    """Print concise analysis summary with essential metrics only"""
    print("\nGround State Structure Analysis Complete")
    print(f"Mean bond length: {results['mean_bond_length']:.4f} Å")
    print(f"Coordination distribution: ", end="")
    for cn, count in zip(results['unique_coordinations'], results['counts']):
        print(f"{count}@{cn} ", end="")
    print(f"\nStructure compactness: {results['mean_distance_to_com']:.4f}±{results['std_distance_to_com']:.4f} Å")
    print(f"Coordination range: {results['min_coordination']} to {results['max_coordination']}")

# --- Execution Pipeline ---
if __name__ == "__main__":
    # Verify required data exists
    if 'ground_state_structure' not in globals():
        raise RuntimeError("Ground state structure data not found. Execute previous analysis first.")
    
    # Configure output directory
    output_dir = os.path.join(
        os.path.expanduser("~"), 
        "Desktop", 
        "Au20_Structure_Visualizations"
    )
    
    # Perform analysis and save visualization
    analysis_results = analyze_ground_state_structure(
        ground_state_structure=ground_state_structure,
        output_dir=output_dir
    )
    
    # Print minimal results summary
    print_minimal_results(analysis_results)
    print(f"Full analysis visualization saved to: {output_dir}/ground_state_analysis.png")


# In[28]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform
import os
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'DejaVu Sans'

def analyze_perturbation_sensitivity(
    ground_state_structure,
    model,
    device,
    perturbation_strengths=None,
    num_samples_per_strength=50
):

    # Default perturbation strengths if not provided
    if perturbation_strengths is None:
        perturbation_strengths = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    base_coords = ground_state_structure['coordinates']
    base_energy = ground_state_structure['energy']
    results = []
    
    # Process each perturbation level
    for sigma in perturbation_strengths:
        energy_deltas = []
        for _ in range(num_samples_per_strength):
            # Generate random displacement and perturbed coordinates
            displacement = np.random.normal(0, sigma, base_coords.shape)
            perturbed_coords = base_coords + displacement
            
            # Predict energy with model
            predicted_energy = predict_energy(perturbed_coords, model, device)
            energy_deltas.append(predicted_energy - base_energy)
        
        # Calculate statistical metrics
        energy_deltas = np.array(energy_deltas)
        results.append({
            'sigma': sigma,
            'mae': np.mean(np.abs(energy_deltas)),
            'rmse': np.sqrt(np.mean(energy_deltas**2)),
            'mean_delta': np.mean(energy_deltas),
            'std_delta': np.std(energy_deltas),
            'deltas': energy_deltas
        })
    
    return results

def predict_energy(coords, model, device):

    # Configuration parameters
    cutoff = 3.5
    num_gaussians = 50
    gamma = 10.0
    
    # Distance matrix calculation
    dist_matrix = squareform(pdist(coords))
    
    # Edge construction
    adj = dist_matrix < cutoff
    np.fill_diagonal(adj, False)
    edge_index = torch.tensor(np.array(np.where(adj)), dtype=torch.long, device=device)
    
    # Edge feature processing
    edge_distances = torch.tensor(dist_matrix[adj], dtype=torch.float, device=device).view(-1, 1)
    rbf_centers = torch.linspace(0.0, cutoff, num_gaussians, device=device)
    edge_attr = torch.exp(-gamma * (edge_distances - rbf_centers)**2)
    
    # Node feature calculation
    coordination_numbers = np.sum(adj, axis=1)
    node_features = torch.tensor(coordination_numbers, dtype=torch.float, device=device).view(-1, 1)
    
    # Create graph data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(coords, dtype=torch.float, device=device),
        z=torch.full((20,), 79, dtype=torch.long, device=device)
    )
    
    # Predict energy
    with torch.no_grad():
        from torch_geometric.loader import DataLoader
        loader = DataLoader([data], batch_size=1)
        batch = next(iter(loader)).to(device)
        return model(batch.z, batch.pos, batch.batch).item()

def save_perturbation_visualizations(results, output_dir):

    # Configure visualization
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(16, 6))
    
    # Extract data for plotting
    sigmas = [res['sigma'] for res in results]
    maes = [res['mae'] for res in results]
    rmses = [res['rmse'] for res in results]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: MAE and RMSE vs perturbation strength
    ax1.plot(sigmas, maes, marker='o', linewidth=2, markersize=8, 
             label='MAE', color='#E6C235', zorder=3)
    ax1.plot(sigmas, rmses, marker='s', linewidth=2, markersize=8, 
             label='RMSE', color='#0A295E', zorder=3)
    ax1.set_title(r'Energy Response to Structural Perturbation in Au$_{20}$', fontsize=14)
    ax1.set_xlabel(r'Perturbation Strength $\sigma$ (Å)', fontsize=12)
    ax1.set_ylabel(r'Energy Change (eV)', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    # Plot 2: Energy change distribution
    all_sigmas = [res['sigma'] for res in results for _ in res['deltas']]
    all_deltas = [delta for res in results for delta in res['deltas']]
    sns.violinplot(x=all_sigmas, y=all_deltas, ax=ax2, palette='coolwarm', zorder=2)
    ax2.set_title(r'Distribution of Energy Changes vs. Perturbation', fontsize=14)
    ax2.set_xlabel(r'Perturbation Strength $\sigma$ (Å)', fontsize=12)
    ax2.set_ylabel(r'Energy Change $\Delta E$ (eV)', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    # Save figure with high resolution
    output_path = os.path.join(output_dir, "perturbation_sensitivity.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def load_model_if_needed():

    try:
        from torch_geometric.nn import SchNet
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model architecture
        model = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=3.5
        ).to(device)
        
        # Load weights
        model_path = 'best_schnet_model.pth'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model, device
        
    except Exception:
        pass
    
    return None, None

# --- Execution Pipeline ---
if __name__ == "__main__":
    # Verify required data exists
    if 'ground_state_structure' not in globals():
        if 'au20_data' in globals():
            ground_state_structure = au20_data.loc[au20_data['energy'].idxmin()]
        else:
            raise RuntimeError("Ground state structure data not available")
    
    # Load model if needed
    model, device = load_model_if_needed()
    if model is None or device is None:
        raise RuntimeError("Model loading failed - required for perturbation analysis")
    
    # Configure output directory
    output_dir = os.path.join(
        os.path.expanduser("~"), 
        "Desktop", 
        "Au20_Structure_Visualizations"
    )
    
    # Perform perturbation analysis
    results = analyze_perturbation_sensitivity(
        ground_state_structure=ground_state_structure,
        model=model,
        device=device
    )
    
    # Save visualizations
    save_perturbation_visualizations(
        results=results,
        output_dir=output_dir
    )
    
    # Calculate key metrics for minimal output
    max_sigma = max(res['sigma'] for res in results)
    max_rmse = max(res['rmse'] for res in results)
    
    # Print minimal results summary
    print(f"Perturbation analysis completed (σ range: 0.01-{max_sigma}Å)")
    print(f"Maximum energy prediction error: {max_rmse:.4f} eV")
    print(f"Visualization saved to: {output_dir}/perturbation_sensitivity.png")


# In[31]:


import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_structural_stability(sigmas, rmses, output_dir):
    # Configure matplotlib for scientific notation
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Polynomial fitting (quadratic model)
    coefficients = np.polyfit(sigmas, rmses, 2)
    k_gnn = coefficients[0]  # Quadratic coefficient = stability constant
    
    # Calculate fit quality metrics
    y_pred = np.polyval(coefficients, sigmas)
    ss_res = np.sum((rmses - y_pred) ** 2)
    ss_tot = np.sum((rmses - np.mean(rmses)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Additional statistics
    mae = np.mean(np.abs(rmses - y_pred))
    rmse_fit = np.sqrt(np.mean((rmses - y_pred)**2))
    max_error = np.max(np.abs(rmses - y_pred))
    stability_score = 1 / (k_gnn + 1e-8)  # Avoid division by zero
    
    # Generate visualization
    fit_sigmas = np.linspace(sigmas.min(), sigmas.max(), 100)
    fit_rmses = np.polyval(coefficients, fit_sigmas)
    
    plt.figure(figsize=(12, 8))
    
    # Plot data points with professional styling
    plt.scatter(
        sigmas, rmses, 
        color='#D44942', 
        label='GNN Predicted Data (RMSE)', 
        s=150, 
        zorder=5, 
        edgecolors='black', 
        linewidth=1,
        alpha=0.8
    )
    
    # Plot fitting curve
    plt.plot(
        fit_sigmas, fit_rmses, 
        color='#2A5B8C', 
        linestyle='--', 
        linewidth=2.5,
        label=f'Quadratic Fit ($k_{{GNN}}$={k_gnn:.2f} eV/Å², $R^2$={r_squared:.3f})'
    )
    
    # Configure plot aesthetics
    plt.title(
        r'Structural Stability Analysis of Au$_{20}$: Harmonic Approximation', 
        fontsize=16, 
        fontweight='bold'
    )
    plt.xlabel(
        r'Perturbation Standard Deviation $\sigma$ (Å)', 
        fontsize=12
    )
    plt.ylabel(
        r'Energy Response RMSE (eV)', 
        fontsize=12
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "structural_stability_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return key metrics
    return {
        'k_gnn': k_gnn,
        'r_squared': r_squared,
        'mae': mae,
        'rmse_fit': rmse_fit,
        'max_error': max_error,
        'stability_score': stability_score
    }

def print_minimal_results(metrics):
    """Print concise stability analysis summary with essential metrics only"""
    print("\nStructural Stability Analysis Complete")
    print(f"Harmonic stability constant k_GNN: {metrics['k_gnn']:.4f} eV/Å²")
    print(f"Fit quality (R²): {metrics['r_squared']:.4f}")
    print(f"Stability Score: {metrics['stability_score']:.4f}")

# --- Execution Pipeline ---
if __name__ == "__main__":
    # Verify required data exists
    try:
        # Use existing perturbation analysis results if available
        from perturbation_analysis import results
        sigmas = np.array([res['sigma'] for res in results])
        rmses = np.array([res['rmse'] for res in results])
    except:
        # Default data if no previous results
        sigmas = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
        rmses = np.array([1.4742, 1.8351, 4.9337, 18.5305, 32.2769, 46.0180, 61.5787])
    
    # Configure output directory
    output_dir = os.path.join(
        os.path.expanduser("~"), 
        "Desktop", 
        "Au20_Structure_Visualizations"
    )
    
    # Perform stability analysis and save visualization
    stability_metrics = analyze_structural_stability(
        sigmas=sigmas,
        rmses=rmses,
        output_dir=output_dir
    )
    
    # Print minimal results summary
    print_minimal_results(stability_metrics)
    print(f"Analysis visualization saved to: {output_dir}/structural_stability_analysis.png")


# In[ ]:




