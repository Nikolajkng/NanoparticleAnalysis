import matplotlib.pyplot as plt

def plotLoss(lossValues: list[float]) -> None:
    plt.yscale('log')
    plt.plot(lossValues, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.pause(0.1)  # pause to update the plot