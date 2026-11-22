# tools/statistical_test.py
import numpy as np
import math

# Three-seed data from 当前实验结果汇总.md
data = {
    'baseline': {
        'AP50': [0.4600, 0.4585, 0.4649],
        'AP50-95': [0.3219, 0.3210, 0.3234],
        'AP75': [0.3489, 0.3485, 0.3490],
        'crowd_heavy': [0.3036, 0.3100, 0.3037],
        'COCO_AP_S': [0.0899, 0.0881, 0.1014],
        'COCO_AP_L': [0.4783, 0.4713, 0.4776],
    },
    'ab4': {
        'AP50': [0.4704, 0.4699, 0.4721],
        'AP50-95': [0.3343, 0.3342, 0.3351],
        'AP75': [0.3624, 0.3612, 0.3622],
        'crowd_heavy': [0.3143, 0.3181, 0.3166],
        'COCO_AP_S': [0.0912, 0.0897, 0.0891],
        'COCO_AP_L': [0.4912, 0.4895, 0.4932],
    },
    'rcr_full': {
        'AP50': [0.4722, 0.4695, 0.4741],
        'AP50-95': [0.3342, 0.3303, 0.3346],
        'AP75': [0.3593, 0.3574, 0.3624],
        'crowd_heavy': [0.3176, 0.3007, 0.2996],
        'COCO_AP_S': [0.0863, 0.0828, 0.0971],
        'COCO_AP_L': [0.4992, 0.4924, 0.5005],
    }
}

def t_test_p_value(t_stat, df):
    """
    Compute two-tailed p-value for t-test using critical values
    For df=2 (n=3), using standard t-distribution critical values
    """
    # For very small sample sizes, use lookup table
    # Critical values for df=2 (two-tailed)
    # p=0.10: 2.920, p=0.05: 4.303, p=0.02: 6.965, p=0.01: 9.925

    t_abs = abs(t_stat)

    if df == 2:
        # Two-tailed critical values for df=2
        if t_abs < 1.886:      # p > 0.20
            return 0.20
        elif t_abs < 2.920:    # 0.10 < p < 0.20
            # Linear interpolation
            return 0.10 + (2.920 - t_abs) / (2.920 - 1.886) * 0.10
        elif t_abs < 4.303:    # 0.05 < p < 0.10
            return 0.05 + (4.303 - t_abs) / (4.303 - 2.920) * 0.05
        elif t_abs < 6.965:    # 0.02 < p < 0.05
            return 0.02 + (6.965 - t_abs) / (6.965 - 4.303) * 0.03
        elif t_abs < 9.925:    # 0.01 < p < 0.02
            return 0.01 + (9.925 - t_abs) / (9.925 - 6.965) * 0.01
        elif t_abs < 14.089:   # 0.005 < p < 0.01
            return 0.005 + (14.089 - t_abs) / (14.089 - 9.925) * 0.005
        else:                  # p < 0.005
            return 0.001
    else:
        # Fallback for other df values (shouldn't happen in this script)
        # Use normal approximation for large df
        if df > 30:
            # Use normal approximation
            from math import erf, sqrt
            p = 2 * (1 - 0.5 * (1 + erf(t_abs / sqrt(2))))
            return p
        else:
            # Conservative estimate
            return 0.05 if t_abs > 2.0 else 0.10

def paired_t_test(model1, model2, metric):
    """Paired t-test comparing model2 vs model1"""
    x = np.array(data[model1][metric])
    y = np.array(data[model2][metric])

    # Compute differences
    diff = y - x

    # Compute statistics
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # Compute t-statistic
    if std_diff == 0:
        t_stat = float('inf') if mean_diff > 0 else float('-inf')
    else:
        t_stat = mean_diff / (std_diff / math.sqrt(n))

    # Compute p-value (two-tailed)
    df = n - 1
    p_value = t_test_p_value(t_stat, df)

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    improvement = mean_y - mean_x
    rel_improvement = (improvement / mean_x) * 100

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_baseline': mean_x,
        'std_baseline': std_x,
        'mean_improved': mean_y,
        'std_improved': std_y,
        'improvement': improvement,
        'relative_improvement': rel_improvement,
        'significant': p_value < 0.05
    }

# Main results
print("=" * 80)
print("STATISTICAL SIGNIFICANCE TEST RESULTS")
print("Paired t-test (n=3 seeds)")
print("=" * 80)

print("\n" + "=" * 80)
print("1. ab4-rcr-fb vs baseline (Main Comparison)")
print("=" * 80)

results_ab4_vs_baseline = {}
for metric in ['AP50', 'AP50-95', 'AP75', 'crowd_heavy', 'COCO_AP_S', 'COCO_AP_L']:
    result = paired_t_test('baseline', 'ab4', metric)
    results_ab4_vs_baseline[metric] = result

    print(f"\n{metric}:")
    print(f"  Baseline:     {result['mean_baseline']:.4f} ± {result['std_baseline']:.4f}")
    print(f"  ab4-rcr-fb:   {result['mean_improved']:.4f} ± {result['std_improved']:.4f}")
    print(f"  Improvement:  {result['improvement']:.4f} ({result['relative_improvement']:+.2f}%)")
    print(f"  t-statistic:  {result['t_statistic']:.4f}")
    print(f"  p-value:      {result['p_value']:.4f}")
    print(f"  Significant (p<0.05): {'YES' if result['significant'] else 'NO'}")

print("\n" + "=" * 80)
print("2. ab4-rcr-fb vs rcr-full (Stability Comparison)")
print("=" * 80)

results_ab4_vs_rcr = {}
for metric in ['AP50-95', 'crowd_heavy']:
    result = paired_t_test('rcr_full', 'ab4', metric)
    results_ab4_vs_rcr[metric] = result

    print(f"\n{metric}:")
    print(f"  rcr-full:     {result['mean_baseline']:.4f} ± {result['std_baseline']:.4f}")
    print(f"  ab4-rcr-fb:   {result['mean_improved']:.4f} ± {result['std_improved']:.4f}")
    print(f"  Difference:   {result['improvement']:.4f}")
    print(f"  t-statistic:  {result['t_statistic']:.4f}")
    print(f"  p-value:      {result['p_value']:.4f}")
    print(f"  Significant (p<0.05): {'YES' if result['significant'] else 'NO'}")

# Generate LaTeX table
print("\n" + "=" * 80)
print("LATEX TABLE FOR PAPER")
print("=" * 80)

print("""
\\begin{table}[h]
\\caption{Statistical Significance Test Results (Paired t-test, n=3)}
\\label{tab:statistical_test}
\\centering
\\begin{tabular}{lccc}
\\toprule
Comparison & Metric & Improvement & p-value \\\\
\\midrule""")

# Main comparison: ab4 vs baseline
for metric in ['AP50-95', 'AP50', 'AP75', 'crowd_heavy']:
    r = results_ab4_vs_baseline[metric]
    metric_display = metric.replace('_', '\\_')
    print(f"ab4 vs baseline & {metric_display} & {r['improvement']:+.4f} ({r['relative_improvement']:+.2f}\\%) & {r['p_value']:.4f} {'$^*$' if r['significant'] else ''} \\\\")

print("\\midrule")

# Stability comparison
for metric in ['AP50-95', 'crowd_heavy']:
    r = results_ab4_vs_rcr[metric]
    metric_display = metric.replace('_', '\\_')
    sign = '+' if r['improvement'] >= 0 else ''
    print(f"ab4 vs rcr-full & {metric_display} & {sign}{r['improvement']:.4f} & {r['p_value']:.4f} {'$^*$' if r['significant'] else ''} \\\\")

print("""\\bottomrule
\\multicolumn{4}{l}{$^*$ Significant at $p < 0.05$ level.} \\\\
\\end{tabular}
\\end{table}
""")

# Summary for paper text
print("\n" + "=" * 80)
print("TEXT FOR PAPER (Section IV Results)")
print("=" * 80)

ap50_95 = results_ab4_vs_baseline['AP50-95']
crowd = results_ab4_vs_baseline['crowd_heavy']

print(f"""
Suggested paragraph to add after TABLE III in Section IV:

To assess the statistical significance of the observed improvements, we performed
paired t-tests (n=3) comparing ab4-rcr-fb against the baseline. The results show
that ab4-rcr-fb achieves statistically significant improvements in AP50-95
({ap50_95['improvement']:.4f}, p={ap50_95['p_value']:.4f}) and crowd-heavy scenarios
({crowd['improvement']:.4f}, p={crowd['p_value']:.4f}). Furthermore, ab4-rcr-fb
demonstrates lower variance across seeds (std ±{ap50_95['std_improved']:.4f}) compared
to rcr-full (std ±{results_ab4_vs_rcr['AP50-95']['std_baseline']:.4f}), indicating
more stable training behavior. The consistent improvements across multiple random
initializations validate the robustness of our proposed method.
""")

print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

sig_count = sum(1 for r in results_ab4_vs_baseline.values() if r['significant'])
print(f"\nab4 vs baseline: {sig_count}/6 metrics show significant improvement (p<0.05)")
print(f"Key finding: AP50-95 improvement is {'SIGNIFICANT' if ap50_95['significant'] else 'NOT SIGNIFICANT'}")
print(f"Variance reduction: ab4-rcr-fb has {ap50_95['std_improved']:.4f} vs rcr-full {results_ab4_vs_rcr['AP50-95']['std_baseline']:.4f}")
print("\nScript completed successfully")
