import numpy as np 
import matplotlib.pyplot as plt

#1. taking any distribution 
#drawing samples from it --> Compute the mean of the samples 
# as the sample size grows, the means form a normal distribution 
# This is the central limit theorem 

def demostrate_clt(
    sample_size: int,
    n_experiments: int = 10_000
)-> np.ndarray:
    """
    Draw n_experiments samples of size sample_size from an
    exponential distribution (heavily skewed — not normal at all).
    Return the distribution of sample means.
    """
    means = []
    for _ in range(n_experiments):
        sample = np.random.exponential(scale=1.0, size=sample_size)
        means.append(np.mean(sample))
    return np.array(means)

def plot_clt():
    """
    Show how sample mean distribution changes as sample size grows.
    Small n = skewed. LArge n = Normal. Tht's CLT 
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    sample_sizes = [1, 5,  30, 300]

    for ax, n in zip(axes, sample_sizes):
        means = demostrate_clt(sample_size=n)
        ax.hist(means, bins=60, edgecolor='black', color='steelblue', alpha=0.7)
        ax.set_title(f"Sample size n={n}")
        ax.set_xlabel("Sample mean")
        ax.set_ylabel("Frequency")
        ax.axvline(np.mean(means), color='red', linestyle='--', label=f'mean={np.mean(means):.2f}')
        ax.legend()

    plt.suptitle(
        "Central Limit Theorem - Exponential distribution\n"
        "Each plot: distribution of 10,000 sample means",
        fontsize = 13
    )
    plt.tight_layout()
    plt.savefig("notebooks/clt_demo.png", dpi=150)
    print("Plot saved to notebooks/clt_demo.png")

# ── Why this matters for ML ────────────────────────────────────────────────────
# When we compute model accuracy on a test set, that accuracy is a sample mean.
# CLT tells us it's approximately normally distributed.
# Which means we can compute a confidence interval around it.

def model_accuracy_confidence_interval(
    n_correct: int,
    n_total: int,
    confidence: float = 0.95
) -> tuple[float, float]:
    """
    Your model got n_correct right out of n_total.
    What's the true accuracy range you can claim?

    Uses normal approximation (valid because of CLT when n is large).
    """
    accuracy = n_correct / n_total
    std_error = np.sqrt(accuracy * (1 - accuracy) / n_total)

    # z-score for 95% confidence = 1.96
    z = 1.96 if confidence == 0.95 else 2.576  # 2.576 for 99%

    lower = accuracy - z * std_error
    upper = accuracy + z * std_error
    return lower, upper


if __name__ == "__main__":
    print("=== Central Limit Theorem ===")
    print("Generating plot... (4 panels showing CLT in action)")
    plot_clt()

    print("\n=== Why CLT matters for your ML models ===")

    # Scenario 1: Small test set
    lower, upper = model_accuracy_confidence_interval(
        n_correct=85, n_total=100
    )
    print(f"\n  Test set: 100 samples, 85 correct")
    print(f"  Accuracy: 85.0%")
    print(f"  95% CI:   ({lower:.1%}, {upper:.1%})")
    print(f"  Honest claim: accuracy is somewhere between {lower:.1%} and {upper:.1%}")

    # Scenario 2: Large test set
    lower, upper = model_accuracy_confidence_interval(
        n_correct=8500, n_total=10_000
    )
    print(f"\n  Test set: 10,000 samples, 8,500 correct")
    print(f"  Accuracy: 85.0%")
    print(f"  95% CI:   ({lower:.1%}, {upper:.1%})")
    print(f"  Honest claim: accuracy is somewhere between {lower:.1%} and {upper:.1%}")

    print("\n  Lesson: same accuracy, completely different confidence.")
    print("  A model evaluated on 100 samples tells you almost nothing.")
    print("  This is what EvalVarianceCheck in MLSanity is actually measuring.")