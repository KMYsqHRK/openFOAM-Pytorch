import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CFDDataset(Dataset):
    def __init__(self, wall_speed, pressure):
        self.wall_speed = wall_speed
        self.pressure = pressure
    
    def __len__(self):
        return len(self.wall_speed)
    
    def __getitem__(self, idx):
        return self.wall_speed[idx], self.pressure[idx]

class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 400)
        )
    
    def forward(self, x):
        return self.network(x)

def train_surrogate_model(wall_speed, pressure, epochs=1000, save_path='surrogate_model.pth'):
    # データの準備とGPUへの移動
    wall_speed = wall_speed.reshape(-1, 1).float().to(device)
    pressure = pressure.float().to(device)
    
    # データセットとデータローダーの作成
    dataset = CFDDataset(wall_speed, pressure)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # モデルの初期化とGPUへの移動
    model = SurrogateModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    loss_history = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_wall_speed, batch_pressure in dataloader:
            # データをGPUに移動
            batch_wall_speed = batch_wall_speed.to(device)
            batch_pressure = batch_pressure.to(device)
            
            pred_pressure = model(batch_wall_speed)
            loss = criterion(pred_pressure, batch_pressure)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss/len(dataloader)
        loss_history.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f} (Best model saved)')
        elif (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    return model, loss_history

def visualize_results(model, wall_speed, pressure, loss_history):
    model.eval()
    with torch.no_grad():
        # 予測を行う（GPUで）
        wall_speed_input = wall_speed.reshape(-1, 1).float().to(device)
        predicted_pressure = model(wall_speed_input)
        
        # CPU に戻してnumpy配列に変換
        wall_speed = wall_speed.cpu().numpy()
        pressure = pressure.cpu().numpy()
        predicted_pressure = predicted_pressure.cpu().numpy()
    
    # プロット部分は変更なし
    fig = plt.figure(figsize=(7.5, 6))
    gs = GridSpec(3, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(pressure, aspect='auto', cmap='viridis')
    ax1.set_title('Actual Pressure Distribution')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Time Step')
    plt.colorbar(im1, ax=ax1, label='Pressure')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(predicted_pressure, aspect='auto', cmap='viridis')
    ax2.set_title('Predicted Pressure Distribution')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Time Step')
    plt.colorbar(im2, ax=ax2, label='Pressure')
    
    middle_case = 9
    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(pressure.shape[1])
    ax3.plot(x, pressure[middle_case], label=f'Actual (wall_speed={wall_speed[middle_case]:.2f})', alpha=0.7)
    ax3.plot(x, predicted_pressure[middle_case], label=f'Predicted', alpha=0.7)
    ax3.set_title(f'Pressure Profile at Wall Speed {middle_case}')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Pressure')
    ax3.legend()
    ax3.grid(True)
    
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(loss_history)
    ax4.set_title('Training Loss History')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_yscale('log')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # データの読み込み
    data = torch.load('cavity_data.pt')
    wall_speed = data['wall_speed']
    pressure = data['pressure']
    pressure -= 100000  # 正規化
    
    print("Training model...")
    model, loss_history = train_surrogate_model(wall_speed, pressure)
    
    print("Visualizing results...")
    visualize_results(model, wall_speed, pressure, loss_history)

if __name__ == "__main__":
    main()