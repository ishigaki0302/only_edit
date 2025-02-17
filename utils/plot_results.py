import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
from scipy.stats import t
from statistics import mean
import pickle

# def plot_results(all_old_probs, all_new_probs, all_probs_diff, avg_first_old_probs, avg_first_new_probs, avg_last_old_probs, avg_last_new_probs, output_path):
def plot_results(all_old_probs, all_new_probs, all_probs_diff, all_history_effect_old_probs, all_history_effect_new_probs, output_path):
    # print(f'avg_first_old_probs: {avg_first_old_probs}')
    # print(f'avg_first_new_probs: {avg_first_new_probs}')
    # print(f'avg_last_old_probs: {avg_last_old_probs}')
    # print(f'avg_last_new_probs: {avg_last_new_probs}')
    
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
    # plt.hlines(avg_first_old_probs, np.min(x), np.max(x), colors='red', linestyle='dashed', linewidth=3)
    # plt.hlines(avg_last_old_probs, np.min(x), np.max(x), colors='red', linewidth=3)
    plt.xlabel('update step')
    plt.ylabel('P(o)')
    plt.xticks(x)
    plt.title('Mean Probabilities with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}_old.png")

    # プロットの設定
    # plt.figure(figsize=(10, 6))
    # plt.hlines(avg_first_old_probs, np.min(x), np.max(x), colors='red', linestyle='dashed', linewidth=3)
    # plt.hlines(avg_last_old_probs, np.min(x), np.max(x), colors='red', linewidth=3)
    # plt.xlabel('update step')
    # plt.ylabel('P(o))')
    # plt.xticks(x)
    # plt.title('Mean Probabilities with 95% Confidence Interval')
    # plt.grid(True)
    # plt.savefig(f"{output_path}_old_first_last.png")

    # プロットの設定
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_new_probs, label='New Probs', marker='o')
    plt.fill_between(x, ci_new_probs[0], ci_new_probs[1], alpha=0.2)
    # plt.hlines(avg_first_new_probs, np.min(x), np.max(x), colors='red', linestyle='dashed', linewidth=3)
    # plt.hlines(avg_last_new_probs, np.min(x), np.max(x), colors='red', linewidth=3)
    plt.xlabel('update step')
    plt.ylabel('P(o*)')
    plt.xticks(x)
    plt.title('Mean Probabilities with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}_new.png")

    # プロットの設定
    # plt.figure(figsize=(10, 6))
    # plt.hlines(avg_first_new_probs, np.min(x), np.max(x), colors='red', linestyle='dashed', linewidth=3)
    # plt.hlines(avg_last_new_probs, np.min(x), np.max(x), colors='red', linewidth=3)
    # plt.xlabel('update step')
    # plt.ylabel('P(o*))')
    # plt.xticks(x)
    # plt.title('Mean Probabilities with 95% Confidence Interval')
    # plt.grid(True)
    # plt.savefig(f"{output_path}_new_first_last.png")

    # プロットの設定
    plt.figure(figsize=(10, 6))
    plt.plot(x, mean_probs_diff, label='Probs Diff', marker='o')
    plt.fill_between(x, ci_probs_diff[0], ci_probs_diff[1], alpha=0.2)
    plt.xlabel('update step')
    plt.ylabel('P(o*)) - P(o))')
    plt.xticks(x)
    plt.title('Mean Probabilities with 95% Confidence Interval')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}_diff.png")

    # plt.figure(figsize=(10, 6))
    # for i in range(5):
    #     # 全データの平均を計算
    #     mean_old_probs = np.mean(all_old_probs[i*100:(i+1)*100], axis=0)
    #     # 全データの標準偏差を計算
    #     std_old_probs = np.std(all_old_probs[i*100:(i+1)*100], axis=0)
    #     # 95%信頼区間を計算
    #     ci_old_probs = t.interval(0.95, n-1, loc=mean_old_probs, scale=std_old_probs/np.sqrt(n/5))
    #     # プロットの設定
    #     plt.plot(x, mean_old_probs, label=f'Old Probs (~{(i+1)*100})', marker='o')
    #     plt.fill_between(x, ci_old_probs[0], ci_old_probs[1], alpha=0.2)
    # # plt.hlines(avg_first_old_probs, np.min(x), np.max(x), colors='red', linestyle='dashed', linewidth=3)
    # # plt.hlines(avg_last_old_probs, np.min(x), np.max(x), colors='red', linewidth=3)
    # plt.xlabel('update step')
    # plt.ylabel('P(o))')
    # plt.xticks(x)
    # plt.title('Mean Probabilities with 95% Confidence Interval')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"{output_path}_old_every100.png")

    # plt.figure(figsize=(10, 6))
    # for i in range(5):
    #     # 全データの平均を計算
    #     mean_new_probs = np.mean(all_new_probs[i*100:(i+1)*100], axis=0)
    #     # 全データの標準偏差を計算
    #     std_new_probs = np.std(all_new_probs[i*100:(i+1)*100], axis=0)
    #     # 95%信頼区間を計算
    #     ci_new_probs = t.interval(0.95, n-1, loc=mean_new_probs, scale=std_new_probs/np.sqrt(n/5))
    #     # プロットの設定
    #     plt.plot(x, mean_new_probs, label=f'New Probs (~{(i+1)*100})', marker='o')
    #     plt.fill_between(x, ci_new_probs[0], ci_new_probs[1], alpha=0.2)
    # # plt.hlines(avg_first_new_probs, np.min(x), np.max(x), colors='red', linestyle='dashed', linewidth=3)
    # # plt.hlines(avg_last_new_probs, np.min(x), np.max(x), colors='red', linewidth=3)
    # plt.xlabel('update step')
    # plt.ylabel('P(o*))')
    # plt.xticks(x)
    # plt.title('Mean Probabilities with 95% Confidence Interval')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"{output_path}_new_every100.png")

    if not os.path.exists(f"{output_path}/history_effect_old"):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(f"{output_path}/history_effect_old")
    for i in range(len(all_history_effect_old_probs)):
        plt.figure(figsize=(10, 6))
        plt.scatter(np.arange(len(all_history_effect_old_probs[i])), all_history_effect_old_probs[i])
        plt.xlabel('data index')
        plt.ylabel('P(o*)')
        plt.title('Mean Probabilities with 95% Confidence Interval')
        plt.grid(True)
        plt.savefig(f"{output_path}/history_effect_old/probs_{i}.png")
    if not os.path.exists(f"{output_path}/history_effect_new"):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(f"{output_path}/history_effect_new")
    for i in range(len(all_history_effect_new_probs)):
        plt.figure(figsize=(10, 6))
        plt.scatter(np.arange(len(all_history_effect_new_probs[i])), all_history_effect_new_probs[i])
        plt.xlabel('data index')
        plt.ylabel('P(o)')
        plt.title('Mean Probabilities with 95% Confidence Interval')
        plt.grid(True)
        plt.savefig(f"{output_path}/history_effect_new/probs_{i}.png")


def main():
    num = 500
    method = 1
    file_paths = [f"result/edit_output/EleutherAI_gpt-j-6B/20240526_143248/fill_in_the_blank_format", f"result/edit_output/EleutherAI_gpt-j-6B/20240526_204413/Question_format"]
    # with open(f"{file_path}_first_old.pkl", 'rb') as f:
    #     first_old_probs = pickle.load(f)
    # with open(f"{file_path}_first_new.pkl", 'rb') as f:
    #     first_new_probs = pickle.load(f)
    # with open(f"{file_path}_old.pkl", 'rb') as f:
    #     all_old_probs = pickle.load(f)
    with open(f"{file_paths[method]}_new.pkl", 'rb') as f:
        all_new_probs = pickle.load(f)
    # with open(f"{file_path}_diff.pkl", 'rb') as f:
    #     all_probs_diff = pickle.load(f)
    # with open(f"{file_path}_last_old.pkl", 'rb') as f:
    #     last_old_probs = pickle.load(f)
    # with open(f"{file_path}_last_new.pkl", 'rb') as f:
    #     last_new_probs = pickle.load(f)
    # with open(f"{file_path}_history_effect_old.pkl", 'rb') as f:
    #     all_history_effect_old_probs = pickle.load(f)
    # with open(f"{file_path}_history_effect_new.pkl", 'rb') as f:
    #     all_history_effect_new_probs = pickle.load(f)
    # plt.figure(figsize=(10, 6))
    # plt.scatter(np.arange(len(first_new_probs)), first_new_probs, c = "red")
    # plt.scatter(np.arange(len(last_new_probs)), last_new_probs, c = "blue")
    # plt.savefig(f"{file_path}_new_scatter.png")
    # plot_results(all_old_probs, all_new_probs, all_probs_diff, mean(first_old_probs), mean(first_new_probs), mean(last_old_probs), mean(last_new_probs), f"{file_path}.png")
    # plot_results(all_old_probs, all_new_probs, all_probs_diff, all_history_effect_old_probs, all_history_effect_new_probs, f"{file_path}.png")
    # import pdb;pdb.set_trace()
    methods = ["the method proposed in ROME", "our method"]
    all_new_probs = np.array(all_new_probs)
    print(f"{num}件中95%以上のもの：{np.count_nonzero(all_new_probs[:num, -1] >= 0.95)}")
    print(f"{num}件中95%以上のものも割合：{np.count_nonzero(all_new_probs[:num, -1] >= 0.95) / num}")
    print(f"平均：{np.mean(all_new_probs[:num, -1])}")
    print(f"分散：{np.var(all_new_probs[:num, -1])}")
    plt.figure(figsize=(10, 6))
    plt.hist(all_new_probs[:num, -1], bins=20, range=(0, 1))
    plt.xlabel('P(o*|s,r)')
    plt.ylabel('Number of data')
    plt.title(f'{methods[method]}')
    plt.grid(True)
    plt.ylim(0, 500)
    # plt.ylim(0, 100)
    plt.savefig(f"result/edit_output/EleutherAI_gpt-j-6B/{methods[method]}_hist_{num}.png")

# main()