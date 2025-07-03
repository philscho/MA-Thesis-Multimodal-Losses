import pandas as pd
import matplotlib.pyplot as plt

def plot_results(data, metric='accuracy', title='Model Performance', save_path=None):
    """
    Generate a grouped bar plot for model performance across datasets.

    Parameters:
        data (dict): Nested dictionary of the form:
                     {
                         'ModelA': {'Dataset1': val, 'Dataset2': val, ...},
                         'ModelB': {'Dataset1': val, 'Dataset2': val, ...},
                         ...
                     }
        metric (str): Name of the metric to display on the y-axis (e.g., 'accuracy').
        title (str): Title of the plot.
        save_path (str): If provided, saves the plot to this path.
    """
    # Convert the nested dict to a DataFrame
    df = pd.DataFrame(data).T  # Models as rows
    df = df.sort_index()
    
    # Plot
    ax = df.plot(kind='bar', figsize=(10, 6))
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title)
    ax.set_ylim(0, 1)  # assuming metric is between 0 and 1, adjust if needed
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.legend(title='Datasets', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# Example usage
if __name__ == '__main__':
    results = {
        'Model A': {'CIFAR-10': 0.85, 'ImageNet': 0.65, 'MNIST': 0.98},
        'Model B': {'CIFAR-10': 0.82, 'ImageNet': 0.68, 'MNIST': 0.96},
        'Model C': {'CIFAR-10': 0.88, 'ImageNet': 0.70, 'MNIST': 0.97}
    }

    plot_results(results, metric='accuracy', title='Model Accuracy Comparison')
