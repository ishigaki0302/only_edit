import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
from scipy.stats import t

def plot_results(all_old_probs, all_new_probs, all_probs_diff, output_path):
    # 全データの平均を計算
    mean_old_probs = np.mean(all_old_probs, axis=0)
    mean_new_probs = np.mean(all_new_probs, axis=0)
    mean_probs_diff = np.mean(all_probs_diff, axis=0)

    # 全データの標準偏差を計算
    std_old_probs = np.std(all_old_probs, axis=0)
    std_new_probs = np.std(all_new_probs, axis=0)
    std_probs_diff = np.std(all_probs_diff, axis=0)

    # データ数を取得
    n = len(all_old_probs)

    # 95%信頼区間を計算
    ci_old_probs = t.interval(0.95, n-1, loc=mean_old_probs, scale=std_old_probs/np.sqrt(n))
    ci_new_probs = t.interval(0.95, n-1, loc=mean_new_probs, scale=std_new_probs/np.sqrt(n))
    ci_probs_diff = t.interval(0.95, n-1, loc=mean_probs_diff, scale=std_probs_diff/np.sqrt(n))

    # 横軸の値を設定
    x = np.arange(len(mean_old_probs))

    # プロットの設定
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_old_probs, label='Old Probs', marker='o')
    plt.fill_between(x, ci_old_probs[0], ci_old_probs[1], alpha=0.2)
    plt.xlabel('update step')
    plt.ylabel('P(o*))')
    plt.title('Mean Probabilities with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}_old.png")

    # プロットの設定
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_new_probs, label='New Probs', marker='o')
    plt.fill_between(x, ci_new_probs[0], ci_new_probs[1], alpha=0.2)
    plt.xlabel('update step')
    plt.ylabel('P(o*))')
    plt.title('Mean Probabilities with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}_new.png")

    # プロットの設定
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_probs_diff, label='Probs Diff', marker='o')
    plt.fill_between(x, ci_probs_diff[0], ci_probs_diff[1], alpha=0.2)
    plt.xlabel('update step')
    plt.ylabel('P(o*))')
    plt.title('Mean Probabilities with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}_diff.png")