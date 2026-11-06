import matplotlib.pyplot as plt
def visualize_loss(losses: list, title: str = ""):
    plt.figure(figsize=(9, 4.5))
    plt.plot(range(1, len(losses) + 1), losses, color='#2ca02c', linewidth=2.5, label='Categorical Cross-Entropy Loss')
    plt.title(title if title else "Assignment 2: Training Loss Process (Softmax Regression - NumPy)", fontsize=13, fontweight='bold')
    plt.xlabel("Số vòng lặp (Epochs)", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()