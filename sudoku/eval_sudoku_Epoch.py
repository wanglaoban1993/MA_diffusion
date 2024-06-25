import os
import torch
from glob import glob
from ddsm_original import *
from train_sudoku import *

def load_model(model_path, model_class, device):
    # 初始化模型架构
    model = model_class(define_relative_encoding())
    
    # 加载模型状态字典
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # 设置模型为评估模式
    model.eval()
    model.to(device)
    
    return model

def evaluate_samples(model, sample_shape, batch_size, device, sampler, num_samples=200):
    # 使用 Euler_Maruyama_sampler 生成样本
    samples = sampler(
        model,
        sample_shape,
        batch_size=batch_size,
        max_time=1,
        time_dilation=1,
        num_steps=num_samples,
        random_order=False,
        speed_balanced=False,
        device=device
    )

    # 检查样本是否生成
    if samples is None or len(samples) == 0:
        print("No samples generated.")
        return None
    
    # 评估生成的样本
    accuracy = sudoku_acc(samples)
    if accuracy is None:
        print("sudoku_acc returned None.")
        return None
    # 评估生成的样本
    accuracy = sudoku_acc(samples)
    return accuracy

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 获取父目录路径
    parent_dir = '/home/fe/twang/projects/MA_2024'
    #parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  # 获取父目录路径
    print("Parent directory:", parent_dir)
    
    model_files_pattern = os.path.join(parent_dir, "score_model_reflectboundaries_epoch_*.pth")  # 构建通配符路径
    print("Model files pattern:", model_files_pattern)
    
    model_files = glob(model_files_pattern)  # 使用通配符列出所有模型文件
    model_files.sort()  # 确保文件按照顺序加载
    print("Matched model files:", model_files)
    
    if not model_files:
        print("No model files found. Please check the directory and file pattern.")
        return
    
    # model_files = glob("score_model_epoch_*.pth")  # 使用通配符列出所有模型文件
    # model_files.sort()  # 确保文件按照顺序加载

    sample_shape = (9, 9, 9)
    batch_size = 256
    num_samples = 200  # 可以根据需要调整

    results = []

    for model_path in model_files:
        print(f"Evaluating model: {model_path}")
        
        # 加载模型
        model = load_model(model_path, ScoreNet, device)
        
        # 生成样本并评估
        accuracy = evaluate_samples(
            model,
            sample_shape,
            batch_size,
            device,
            sampler=Euler_Maruyama_sampler,
            num_samples=num_samples
        )
        
        print(f"Sudoku accuracy for {model_path}: {accuracy:.2f}%")
        results.append((model_path, accuracy))
    
    # 输出所有模型的评估结果
    for model_path, accuracy in results:
        print(f"Model: {model_path}, Sudoku accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()

