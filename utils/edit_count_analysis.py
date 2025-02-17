import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import japanize_matplotlib
import pickle

# 日本語フォントの設定
jp_font_path = '/Library/Fonts/ヒラギノ角ゴシック W5.ttc'
jp_font = fm.FontProperties(fname=jp_font_path)

step = 10
method = 0
output_path = "result/edit_data/EleutherAI_gpt-j-6B"
# file_paths = [f"result/edit_output/EleutherAI_gpt-j-6B/20240702_180909/fill_in_the_blank_format", f"result/edit_output/EleutherAI_gpt-j-6B/20240701_182347/Question_format"]
file_paths = [f"result/edit_output/EleutherAI_gpt-j-6B/20240528_005446/fill_in_the_blank_format", f"result/edit_output/EleutherAI_gpt-j-6B/20240528_112909/Question_format"]

# with open(f"{file_paths[method]}_old.pkl", 'rb') as f:
#     all_old_probs = np.array(pickle.load(f))
# with open(f"{file_paths[method]}_new.pkl", 'rb') as f:
#     all_new_probs = np.array(pickle.load(f))
with open(f"{file_paths[method]}_history_effect_old.pkl", 'rb') as f:
    history_effect_old = np.array(pickle.load(f))
with open(f"{file_paths[method]}_history_effect_new.pkl", 'rb') as f:
    history_effect_new = np.array(pickle.load(f))

# # ステップごとの割合を計算
# ratios = []
# for i in range(1, len(all_old_probs) + 1):
#     old_subarray = all_old_probs[:i, step-1]
#     new_subarray = all_new_probs[:i, step-1]
#     ratio = np.sum(new_subarray > old_subarray) / i
#     ratios.append(ratio)

# ステップごとの割合を計算
ratios = []
for i in range(len(history_effect_old)):
    old_subarray = history_effect_old[i][:i+1]
    new_subarray = history_effect_new[i][:i+1]
    print(old_subarray.shape)
    ratio = np.sum(new_subarray > old_subarray) / (i+1)
    ratios.append(ratio)

# プロット
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(ratios) + 1), ratios, marker='o', label=f'step数: {step}')
plt.xlabel('編集回数')
plt.ylabel('all_old_probsをall_new_probsが上回っている割合')
plt.title('all_old_probsをall_new_probsが上回っている割合の推移')
plt.ylim(0.5, 1)
plt.grid(True)
plt.legend()
plt.savefig(f"{output_path}/edit_count{method}.pdf", transparent=True)

# # all_old_probsの散布図プロット
# plt.figure(figsize=(10, 6))
# plt.scatter(range(1, len(ratios) + 1), all_new_probs[:, step-1], marker='o', label=f'step数: {step})
# plt.xlabel('編集回数')
# plt.ylabel('all_old_probs')
# plt.ylim(0, 1)
# plt.grid(True)
# plt.legend()
# plt.savefig(f"{output_path}/edit_count{method}_all_new_probs_scatter.pdf", transparent=True)

# # ヒストグラムのプロット
# plt.figure(figsize=(10, 6))
# plt.hist(all_new_probs[:, step-1], bins=20, edgecolor='black', label=f'step数: {step})
# plt.xlabel('値')
# plt.ylabel('頻度')
# plt.title('all_new_probsのヒストグラム')
# plt.grid(True)
# plt.legend()
# plt.savefig(f"{output_path}/edit_count{method}_all_new_probs_hist.pdf", transparent=True)