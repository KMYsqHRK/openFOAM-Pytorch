import os
import shutil
import re
import subprocess
import numpy as np
from pathlib import Path

"""
ファイル名: ExcuteFoam.py
作成者: 神谷弘貴
作成日: 2025-02-05

概要:
    openFOAMのケースを生成し、実行するためのスクリプト

使用方法:
    事前にopenFOAM環境に入る必要があります。
"""

# 以下、実際のコードが続きます

class CavityGenerator:
    def __init__(self, template_case_path, output_base_dir):
        """
        Parameters:
        -----------
        template_case_path : str or Path
            Path to the original cavity case directory
        output_base_dir : str or Path
            Base directory where generated cases will be stored
        """
        self.template_case_path = Path(template_case_path)
        self.output_base_dir = Path(output_base_dir)
        self.original_dir = Path.cwd()
        
    # 0/UのWallSpeedのみを変更する
    def modify_wall_speed(self, u_file_path, speed):
        """Modify the wall speed in U file."""
        with open(u_file_path, 'r') as f:
            content = f.read()
        
        # Replace the velocity value using regex
        new_content = re.sub(
            r'(movingWall\s*\{\s*type\s+fixedValue\s*;\s*value\s+uniform\s*\()(\d+\.?\d*)(\s*0\s*0\)\s*;)',
            rf'\g<1>{speed}\g<3>',
            content
        )
        
        with open(u_file_path, 'w') as f:
            f.write(new_content)
    
    def create_case(self, speed):
        """Create a new case with specified wall speed."""
        # Create case directory name
        case_name = f'cavity_v{speed}'.replace('.', '_')
        case_dir = self.output_base_dir / case_name
        
        # Copy entire case directory
        if case_dir.exists():
            shutil.rmtree(case_dir)
        shutil.copytree(self.template_case_path, case_dir)
        
        # Modify U file
        u_file_path = case_dir / '0' / 'U'
        self.modify_wall_speed(u_file_path, speed)
        
        return case_dir
    
    def generate_cases(self, speeds):
        """Generate multiple cases with different wall speeds."""
        # Create output base directory if it doesn't exist
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        generated_cases = []
        for speed in speeds:
            case_dir = self.create_case(speed)
            self.run_openfoam(case_dir)
            generated_cases.append({
                'speed': speed,
                'case_dir': case_dir
            })
        
        return generated_cases

    # caseのディレクトリに移動してopenFoam実行その後、元の位置に戻る
    def run_openfoam(self, case_dir):
        """Run OpenFOAM simulation for the case."""
        try:
            # Change to case directory
            os.chdir(case_dir)
            print(f"\nRunning simulation in: {case_dir}")
            
            # Run blockMesh
            print("Running blockMesh...")
            subprocess.run(['blockMesh'], check=True)
            
            # Run rhoPimpleFoam
            print("Running rhoPimpleFoam...")
            subprocess.run(['rhoPimpleFoam'], check=True)
            
        finally:
            # 必ず元のディレクトリに戻る
            os.chdir(self.original_dir)

def main():
    # パスの設定（環境に応じて変更してください）
    template_case = "/home/l/lab_project/Cabity_Learn/cavity_original"
    output_dir = "cavity_training_data"
    
    # 生成する速度のリスト
    speeds = np.arange(0.5, 10, 0.5)  # 必要な速度値に調整してください
    
    # ケース生成器の作成
    generator = CavityGenerator(template_case, output_dir)
    
    # ケースの生成
    generated_cases = generator.generate_cases(speeds)
    
    # 結果の表示
    print(f"\nGenerated {len(generated_cases)} cases:")
    for case in generated_cases:
        print(f"Speed: {case['speed']} m/s")
        print(f"Directory: {case['case_dir']}")
        print()
    
    print("\nTo run the cases, use the following commands in each case directory:")
    print("blockMesh")
    print("rhoPimpleFoam")

if __name__ == "__main__":
    main()