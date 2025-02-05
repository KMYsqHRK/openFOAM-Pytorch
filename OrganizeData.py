import numpy as np
import pandas as pd
from pathlib import Path
import re
from typing import List, Dict, Union
import torch

"""
ファイル名: OrganizeData.py
作成者: 神谷弘貴
作成日: 2025-02-05

概要:
    openFOAMのケースを生成し、実行するためのスクリプト

使用方法:
    事前にopenFOAM環境に入る必要があります。
"""

class OpenFOAMDataConverter:
    def __init__(self, base_dir: Union[str, Path]):
        """
        OpenFOAMのデータを機械学習用テンソルに変換するクラス
        
        Parameters:
        -----------
        base_dir : str or Path
            生成されたケースが格納されているベースディレクトリ
        """
        self.base_dir = Path(base_dir)
        self.time_step = "15"  # 15秒時点のデータを使用

    def _parse_velocity_file(self, file_path: Path) -> np.ndarray:
        """
        OpenFOAMのvelocityファイルを解析してnumpy配列に変換
        
        Parameters:
        -----------
        file_path : Path
            velocityファイルのパス
            
        Returns:
        --------
        np.ndarray
            形状(n_points, 3)の速度ベクトル配列
        """
        velocities = []
        with open(file_path, 'r') as f:
            contents = f.read()
            
        # 速度データの部分を抽出
        data_match = re.search(r'internalField\s+nonuniform\s+List<vector>\s+\d+\s*\((.*?)\)\s*;', 
                             contents, re.DOTALL)
        if data_match:
            data_str = data_match.group(1)
            # 各行から速度ベクトルを抽出
            for line in data_str.strip().split('\n'):
                if '(' in line and ')' in line:
                    v = line.strip('() \n').split()
                    velocities.append([float(x) for x in v])
                    
        return np.array(velocities)

    def _parse_pressure_file(self, file_path: Path) -> np.ndarray:
        """
        OpenFOAMのpressureファイルを解析してnumpy配列に変換
        
        Parameters:
        -----------
        file_path : Path
            pressureファイルのパス
            
        Returns:
        --------
        np.ndarray
            形状(n_points,)の圧力配列
        """
        pressures = []
        with open(file_path, 'r') as f:
            contents = f.read()
            
        # 圧力データの部分を抽出
        data_match = re.search(r'internalField\s+nonuniform\s+List<scalar>\s+\d+\s*\((.*?)\)\s*;', 
                             contents, re.DOTALL)
        if data_match:
            data_str = data_match.group(1)
            # 各行から圧力値を抽出
            for line in data_str.strip().split('\n'):
                pressures.append(float(line.strip()))
                    
        return np.array(pressures)
    
    def _get_wall_speed(self, case_dir: Path) -> float:
        """
        ケースディレクトリから壁面速度を取得
        
        Parameters:
        -----------
        case_dir : Path
            ケースディレクトリのパス
            
        Returns:
        --------
        float
            壁面速度
        """
        u_file = case_dir / '0' / 'U'
        with open(u_file, 'r') as f:
            lines = f.readlines()

        boundary_start = -1
        for i, line in enumerate(lines):
            if 'boundaryField' in line:
                boundary_start = i
                break

        current_section = None
        in_moving_wall = False
        moving_wall_data = 0

        for line in lines[boundary_start:]:
            line = line.strip()
        
            if line.startswith('movingWall'):
                in_moving_wall = True
                continue
            
            if in_moving_wall:
                if 'uniform' in line:
                    # uniform (x y z) の形式から値を取得
                    values = line.split('uniform')[1].strip()
                    values = values.strip('()')
                    values = values.split()
                    moving_wall_data = float(values[0])  # X座標の値
                    break
        return moving_wall_data
        
    
    def convert_to_tensors(self) -> Dict[str, torch.Tensor]:
        """
        全ケースの15秒時点のデータをテンソルに変換
        
        Returns:
        --------
        Dict[str, torch.Tensor]
            以下のキーを持つ辞書:
            - 'velocity': 形状(n_cases, n_points, 3)の速度テンソル
            - 'pressure': 形状(n_cases, n_points)の圧力テンソル
            - 'wall_speed': 形状(n_cases,)の壁面速度テンソル
        """
        all_velocities = []
        all_pressures = []
        all_wall_speeds = []
        
        # 全ケースディレクトリを処理
        for case_dir in sorted(self.base_dir.glob('cavity_v*')):
            time_dir = case_dir / self.time_step
            
            if not time_dir.exists():
                print(f"Warning: Time directory {time_dir} not found in case {case_dir}")
                continue
            
            try:
                # 速度と圧力のデータを読み込み
                velocity = self._parse_velocity_file(time_dir / 'U')
                pressure = self._parse_pressure_file(time_dir / 'p')
                wall_speed = self._get_wall_speed(case_dir)
                
                all_velocities.append(velocity)
                all_pressures.append(pressure)
                all_wall_speeds.append(wall_speed)
                
            except Exception as e:
                print(f"Error processing case {case_dir}: {e}")
                continue
        
        if not all_velocities:
            raise ValueError("No valid data found in any case directory")
        
        # numpy配列からPyTorchテンソルに変換(辞書オブジェクト型)
        tensors = {
            'velocity': torch.tensor(np.stack(all_velocities), dtype=torch.float32),
            'pressure': torch.tensor(np.stack(all_pressures), dtype=torch.float32),
            'wall_speed': torch.tensor(all_wall_speeds, dtype=torch.float32)
        }
        
        return tensors

def main():

    converter = OpenFOAMDataConverter("cavity_training_data")
    tensors = converter.convert_to_tensors()
    
    print("\nデータの形状:")
    for key, tensor in tensors.items():
        print(f"{key}: {tensor.shape}")
        
    # データの保存
    torch.save(tensors, "cavity_data.pt")
    print("\nデータを cavity_data.pt として保存しました")

if __name__ == "__main__":
    main()