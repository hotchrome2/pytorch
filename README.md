# pytorch
## Tensor
PyTorchのテンソルは「配列（array）」の一般化された形であり、その次元によってベクトル（1次元）、行列（2次元）、多次元配列（3次元以上） として扱われます。

## DeepONet
https://www.ecomottblog.com/?p=15463

https://kurporateadvisory.jp/articles/tech/PINN.html

## モデルの入力サイズと出力サイズ
```
def __init__(self, in_size, mid_size, out_size):
```
- in_size: モデルに入力されるベクトルの次元数（特徴量数）
- mid_size: 中間層のノード数
- out_size: モデルが最終的に出力すべきベクトルの次元数。分類ならクラス数、回帰なら1
  - y_train.shape[1]と同じにする。つまり正解データの列数

