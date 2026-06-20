import scipy.stats as stats
import numpy as np

def main():
    n = 2709
    k = 55
    h_star = 0.11
    
    # Calculate Chi2 p-values for leverage values
    print("Leverage vs Chi2 p-value:")
    print("Leverage | Chi2 Statistic | p-value")
    print("-" * 45)
    for lev in np.linspace(0, 1.5 * h_star, 10):
        chi2_stat = (n - 1) * lev
        p_val = 1.0 - stats.chi2.cdf(chi2_stat, k)
        print(f"{lev:.4f} | {chi2_stat:14.4f} | {p_val:.6f}")

if __name__ == "__main__":
    main()
