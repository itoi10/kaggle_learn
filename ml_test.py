# データセット読み込み (アヤメのデータ)
from sklearn import datasets

iris = datasets.load_iris()

#              がくの長さ             がくの幅              花びらの長さ           花びらの幅
# 特徴量の名前: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(f"特徴量の名前: {iris.feature_names}")

#            セトサ     バージカラー   バージニカ
# 分類の名前: ['setosa' 'versicolor' 'virginica']
print(f"分類の名前: {iris.target_names}")

print(f"分類の値: {iris.target}\n")


# データを処理しやすいようにデータフレームにする
import pandas as pd

df = pd.DataFrame(iris.data)
# カラム名設定
df.columns = iris.feature_names
# target列追加
df["target"] = iris.target
print(df)

### 品種ごとのがくの幅をヒストグラム描画
import matplotlib.pyplot as plt

# 品種毎にdfを分ける
df0 = df[df["target"] == 0]
df1 = df[df["target"] == 1]
df2 = df[df["target"] == 2]

plt.figure(figsize=(5, 5))
xx = "sepal width (cm)"  # 対象列(がくの幅)
df0[xx].hist(color="b", alpha=0.5)  # 青グラフ
df1[xx].hist(color="r", alpha=0.5)  # 赤グラフ
df2[xx].hist(color="g", alpha=0.5)  # 緑グラフ

plt.xlabel(xx)
plt.show()

# 散布図 TODO
