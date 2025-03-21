import numpy as np
import matplotlib.pyplot as plt
import random

# 行数・列数
n_cases = 500
n_steps = 20

# 初期データはすべて 0 (白)
data = np.zeros((n_cases, n_steps))

# 各行ごとにランダムに 5～10 個のセルを1（黒）に設定
for i in range(n_cases):
    n_black = random.randint(5, 10)  # その行で塗るセル数
    # 重複しないランダムな列番号を取得
    black_indices = random.sample(range(n_steps), n_black)
    data[i, black_indices] = 1

# プロット作成
plt.figure(figsize=(10, 25))
plt.imshow(data, cmap="binary", aspect="auto")
plt.xlabel("編集ステップ")
plt.ylabel("編集事例")
plt.title("ダミーデータ: 編集事例と編集ステップ")
plt.colorbar(label="状態 (0:白, 1:黒)")
plt.show()