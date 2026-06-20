import scipy.stats as stats
import numpy as np

def main():
    n = 2709
    k = 55
    leverage = 0.0769
    h_star = 0.11
    
    # Calculate F-statistic
    F = (leverage * (n - k)) / k
    print(f"F-statistic for leverage {leverage}: {F:.4f}")
    
    # Calculate p-value (probability of obtaining F or larger)
    p_value = 1.0 - stats.f.cdf(F, k, n - k)
    print(f"p-value: {p_value:.6f}")
    
    # Let's test for leverage from 0 to 1.5 * h_star
    print("\nLeverage vs p-value:")
    for lev in np.linspace(0, 1.5 * h_star, 10):
        F_stat = (lev * (n - k)) / k
        p_val = 1.0 - stats.f.cdf(F_stat, k, n - k)
        print(f"Leverage: {lev:.4f} | F: {F_stat:.4f} | p-value: {p_val:.6f}")

if __name__ == "__main__":
    main()
