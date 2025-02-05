import torch
from TrainSurrogateModel import SurrogateModel # モデルクラスのあるファイルをインポート
import matplotlib.pyplot as plt
import numpy as np

def load_and_predict(input_velocity):
    # モデルの読み込み
    model = SurrogateModel()
    model.load_state_dict(torch.load('surrogate_model.pth'))
    model.eval()  # 評価モード

    # 予測の実行
    with torch.no_grad():  # 勾配計算を無効化（推論時は不要）
        wall_speed = torch.tensor([[input_velocity]]).float()
        predicted_pressure = model(wall_speed)
    
    return predicted_pressure

# 使用例
predicted_pressure = load_and_predict(9.4)
predicted_pressure = predicted_pressure.numpy()
print(predicted_pressure.shape)  # 出力形状の確認

x = np.arange(predicted_pressure.shape[1])
plt.plot(x, predicted_pressure.T)
plt.show()
