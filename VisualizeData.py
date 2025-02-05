import torch
import matplotlib.pyplot as plt
import numpy as np

def load_and_visualize_tensors(file_path):
    # .ptファイルの読み込み
    tensors = torch.load(file_path)
    
    # データの可視化
    plt.figure(figsize=(12, 6))
    
    # 各テンソルのデータをプロット
    for name, tensor in tensors.items():
        # CPU上のNumPy配列に変換
        data = tensor.cpu().numpy()
        
        # データが2次元以上の場合は最初の次元でプロット
        if len(data.shape) > 1:
            plt.plot(data[:, 0], label=f'{name} (first component)')
        else:
            plt.plot(data, label=name)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Tensor Data from .pt File')
    plt.legend()
    plt.grid(True)
    plt.show()

# 使用例
file_path = "cavity_data.pt"
load_and_visualize_tensors(file_path)

# 個別のテンソルの詳細表示
def plot_tensor_details(file_path, tensor_name):
    tensors = torch.load(file_path)
    if tensor_name not in tensors:
        print(f"Tensor '{tensor_name}' not found in file")
        return
        
    data = tensors[tensor_name].cpu().numpy()
    data -= 100000 # すべての計測点から100000を引いている
    
    plt.figure(figsize=(10, 5))
    
    if len(data.shape) > 1:
        for i in range(data.shape[0]):
            plt.plot(data[i, :], label=f'Component {i+1}')
    else:
        plt.plot(data, label=tensor_name)
        
    plt.title(f'{tensor_name} Data')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # データの基本統計量を表示
    print(f"\n{tensor_name} Statistics:")
    print(f"Shape: {data.shape}")
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Std: {np.std(data):.4f}")
    print(f"Min: {np.min(data):.4f}")
    print(f"Max: {np.max(data):.4f}")

# 使用例
# 特定のテンソルの詳細を見る
plot_tensor_details(file_path, 'pressure')
#plot_tensor_details(file_path, 'velocity')
plot_tensor_details(file_path, 'wall_speed')