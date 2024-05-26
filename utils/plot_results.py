import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
from scipy.stats import t
from statistics import mean
import pickle

def plot_results(all_old_probs, all_new_probs, all_probs_diff, avg_first_old_probs, avg_first_new_probs, avg_last_old_probs, avg_last_new_probs, output_path):
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
    plt.hlines(avg_first_old_probs, np.min(x), np.max(x), colors='red', linestyle='dashed', linewidth=3)
    plt.hlines(avg_last_old_probs, np.min(x), np.max(x), colors='red', linewidth=3)
    plt.xlabel('update step')
    plt.ylabel('P(o*))')
    plt.xticks(x)
    plt.title('Mean Probabilities with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}_old.png")

    # プロットの設定
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_new_probs, label='New Probs', marker='o')
    plt.fill_between(x, ci_new_probs[0], ci_new_probs[1], alpha=0.2)
    plt.hlines(avg_first_new_probs, np.min(x), np.max(x), colors='red', linestyle='dashed', linewidth=3)
    plt.hlines(avg_last_new_probs, np.min(x), np.max(x), colors='red', linewidth=3)
    plt.xlabel('update step')
    plt.ylabel('P(o*))')
    plt.xticks(x)
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
    plt.xticks(x)
    plt.title('Mean Probabilities with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}_diff.png")

def main():
    file_path = f"result/edit_output/rinna_japanese-gpt-neox-3.6b/20240526_115126/japanese_Question_format"
    with open(f"{file_path}_first_old.pkl", 'rb') as f:
        first_old_probs = pickle.load(f)
    with open(f"{file_path}_first_new.pkl", 'rb') as f:
        first_new_probs = pickle.load(f)
    with open(f"{file_path}_old.pkl", 'rb') as f:
        all_old_probs = pickle.load(f)
    with open(f"{file_path}_new.pkl", 'rb') as f:
        all_new_probs = pickle.load(f)
    with open(f"{file_path}_diff.pkl", 'rb') as f:
        all_probs_diff = pickle.load(f)
    with open(f"{file_path}_last_old.pkl", 'rb') as f:
        last_old_probs = pickle.load(f)
    with open(f"{file_path}_last_new.pkl", 'rb') as f:
        last_new_probs = pickle.load(f)
    plot_results(all_old_probs, all_new_probs, all_probs_diff, mean(first_old_probs), mean(first_new_probs), mean(last_old_probs), mean(last_new_probs), f"{file_path}.png")

main()