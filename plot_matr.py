import os
import re
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval  # Safe way to convert the string representation of a Python list


def plot_confusion_matrices_final(folder_path, num_classes=10):
    """
    Reads confusion matrices from text files, cleans the content
    (replaces spaces between numbers with commas), and plots them.
    """
    # Create the axis labels (e.g., [0, 1, ..., 9])
    class_labels = np.arange(num_classes)

    # Regex to find files matching 'confusion_n.txt'
    pattern = re.compile(r'confusion_(\d+)\.txt')

    print(f"Searching for files in: {folder_path}")

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)

        if match:
            file_path = os.path.join(folder_path, filename)
            matrix_id = match.group(1)

            print(f"Processing file: {filename}")

            try:
                # 1. Read the content of the file
                with open(file_path, 'r') as f:
                    matrix_string = f.read().strip()

                # 2. --- CRITICAL FIX: Clean the string content ---
                # Remove newlines/carriage returns for single-line processing
                matrix_string = matrix_string.replace('\n', '').replace('\r', '')

                # Use regex to find one or more spaces that are NOT followed or preceded
                # by a square bracket, and replace them with a comma and a space.
                # This ensures numbers are separated by commas, while preserving the inner list structure.
                matrix_string_cleaned = re.sub(r'(?<=\d)\s+(?=\d)', ', ', matrix_string)

                # Also, clean up any stray spaces/commas right next to brackets if they exist
                matrix_string_cleaned = matrix_string_cleaned.replace('[ ', '[').replace(' ]', ']')

                # 3. Safely convert the cleaned string back into a Python list
                cm_list = literal_eval(matrix_string_cleaned)
                cm = np.array(cm_list)

                # Basic validation: Check if it's a square matrix
                if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
                    print(f"Warning: Matrix in {filename} is not square or 2D. Skipping.")
                    continue

                # 4. Plot the Confusion Matrix (using Matplotlib's built-in plot_confusion_matrix logic)

                # Create a figure and axis for the plot
                fig, ax = plt.subplots(figsize=(10, 10))

                # Display the matrix (heatmap)
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

                # Add a colorbar
                cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.set_ylabel('Probability/Ratio', rotation=-90, va="bottom", fontsize=12)

                # Set the labels for the axes
                ax.set(xticks=class_labels,
                       yticks=class_labels,
                       title=f'Normalized Confusion Matrix (ID: {matrix_id})',
                       ylabel='True Label (Actual Class)',
                       xlabel='Predicted Label (Model Output)')

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                # Loop over data dimensions and create text annotations.
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        # Display two decimal places for normalized matrix
                        text_color = "white" if cm[i, j] > thresh else "black"
                        ax.text(j, i, format(cm[i, j], '.2f'),
                                ha="center", va="center",
                                color=text_color, fontsize=10)

                # Adjust layout to prevent labels from being cut off
                fig.tight_layout()

                # 5. Save the figure
                output_filename = f'confusion_matrix_{matrix_id}.png'
                plt.savefig(output_filename)
                plt.close(fig)  # Close the figure to free up memory

                print(f"Successfully plotted and saved: {output_filename}")

            except FileNotFoundError:
                print(f"Error: File not found at {file_path}")
            except ValueError as e:
                # Catches errors from literal_eval if the string isn't parsable even after cleaning
                print(f"Error: Could not parse content of {filename}. Details: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")


# --- Configuration ---
# You need to replace this with the actual path to your folder
folder_of_matrices = './experiments/Baseline_Default_Config0/results/'

# --- Execute the function ---
if __name__ == "__main__":
    # Note: If your folder doesn't exist, this setup will not run, which is fine.
    # Make sure you update 'folder_of_matrices' to the correct location.

    # We infer num_classes=10 from the 10x10 matrix you provided.
    plot_confusion_matrices_final(folder_of_matrices, num_classes=10)

    print("\nScript finished.")
    print("Check the current directory for the saved confusion matrix images (.png files).")