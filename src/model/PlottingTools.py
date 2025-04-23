import matplotlib.pyplot as plt
import numpy as np

def plot_loss(training_loss_values: list[float], validation_loss_values: list[float]) -> None:
    try:
        plt.clf()
        plt.yscale('log')
        plt.plot(training_loss_values, label='Training Loss', color='blue', linestyle='-')
        plt.plot(validation_loss_values, label='Validation Loss', color='red', linestyle='--')
        plt.xticks(np.arange(len(training_loss_values)), np.arange(1, len(training_loss_values)+1))
        plt.legend(loc="upper right")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.pause(0.1)  # pause to update the plot
    except Exception as e:
        print("Error when plotting!")
        print(e.with_traceback)