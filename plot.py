import json
import argparse
import matplotlib.pyplot as plt
import os

# === reference : ChatGPT === #
def plot_training_process(json_file):
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    epochs = list(range(1, len(data["Exact Match"]) + 1))
    
    # Create output directory if it doesn't exist
    output_dir = "plot_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot Exact Match
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, data["Exact Match"], label="Exact Match", marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Exact Match Score')
    plt.title('Exact Match Score over Epochs')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'exact_match_plot.png'))
    plt.close()

    # Plot Training Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, data["Training Loss"], label="Training Loss", marker='o', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss_plot.png'))
    plt.close()

    # Plot Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, data["Validation Loss"], label="Validation Loss", marker='o', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss over Epochs')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'validation_loss_plot.png'))
    plt.close()

    print(f"Plots saved in the '{output_dir}' folder.")

if __name__ == "__main__":
    # Argument parser for file path
    parser = argparse.ArgumentParser(description="Plot training process from JSON file.")
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing training data.')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Plot and save figures
    plot_training_process(args.json_file)
