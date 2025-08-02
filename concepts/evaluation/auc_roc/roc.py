import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc(y_true, y_scores):
    # Compute False Positive Rate, True Positive Rate
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Compute Area Under Curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
