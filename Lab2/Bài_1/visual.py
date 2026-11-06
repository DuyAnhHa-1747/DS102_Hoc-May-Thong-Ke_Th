import matplotlib.pyplot as plt
def visualize_loss(losses: list, title: str = ""):
    plt.figure(figsize=(9, 4.5))
    plt.plot(range(1, len(losses) + 1), losses, color='#1f77b4', linewidth=2.5, label='Training Loss')
    plt.title(title, fontsize=13, fontweight='bold')
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

# Cách gọi trong main.py:
# visualize_loss(lr_model.losses, title="Loss Curve - Logistic Regression")