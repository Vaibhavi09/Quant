import numpy as np 

#__1. Simulate a basic probability space 
# ROlling a fair die 100,000 times and verifying probabilities empirically 

def simulate_die_rolls(n: int = 100_000) -> dict:
    """Roll a fair 6-sided die n times. Return observed probabilities."""
    rolls = np.random.randint(1, 7, size=n)
    probs = {face: np.sum(rolls == face) / n for face in range(1, 7)}
    return probs

# ── 2. Conditional probability ─────────────────────────────────────────────────
# P(A|B) = P(A ∩ B) / P(B)
# Question: given the roll is even, what's the probability it's a 4?

def conditional_prob_die(n: int = 100_000) -> float:
    """P(roll == 4 | roll is even) — should be ~0.333"""
    rolls = np.random.randint(1, 7, size=n)
    even = rolls[rolls % 2 == 0]       # filter to even rolls only
    prob = np.sum(even == 4) / len(even)
    return prob

# ── 3. Bayes' theorem ──────────────────────────────────────────────────────────
# Classic medical test example
# Disease prevalence: 1% of population
# Test sensitivity: 99% (P(positive | disease))
# Test specificity: 95% (P(negative | no disease))
# Question: if you test positive, what's the actual probability you have the disease?

def bayes_disease_test() -> float:
    """
    P(disease | positive) = P(positive | disease) * P(disease)
                            ─────────────────────────────────────
                                      P(positive)
    """
    p_disease = 0.01          # prior — 1% prevalence
    p_pos_given_disease = 0.99     # sensitivity
    p_pos_given_no_disease = 0.05  # 1 - specificity

    p_positive = (p_pos_given_disease * p_disease +
                  p_pos_given_no_disease * (1 - p_disease))

    p_disease_given_pos = (p_pos_given_disease * p_disease) / p_positive
    return p_disease_given_pos

# ── 4. Linearity of expectation ────────────────────────────────────────────────
# E[X + Y] = E[X] + E[Y] — always, even if X and Y are dependent
# Verify with simulation

def linearity_of_expectation(n: int = 100_000) -> dict:
    """
    Roll two dice. Show E[sum] = E[die1] + E[die2] = 3.5 + 3.5 = 7
    """
    die1 = np.random.randint(1, 7, size=n)
    die2 = np.random.randint(1, 7, size=n)

    return {
        "E[die1]": np.mean(die1),
        "E[die2]": np.mean(die2),
        "E[die1] + E[die2]": np.mean(die1) + np.mean(die2),
        "E[die1 + die2]": np.mean(die1 + die2),   # should match above
    }

# ── Run everything ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Die Roll Probabilities ===")
    for face, prob in simulate_die_rolls().items():
        print(f"  P(roll={face}): {prob:.4f}  (expected: 0.1667)")

    print("\n=== Conditional Probability ===")
    print(f"  P(4 | even): {conditional_prob_die():.4f}  (expected: 0.3333)")

    print("\n=== Bayes' Theorem — Medical Test ===")
    result = bayes_disease_test()
    print(f"  P(disease | positive test): {result:.4f}  (expected: ~0.1667)")
    print(f"  Intuition: even with a 99% accurate test, only ~16% chance")
    print(f"  you actually have the disease. This is why Bayes matters.")

    print("\n=== Linearity of Expectation ===")
    for label, val in linearity_of_expectation().items():
        print(f"  {label}: {val:.4f}")