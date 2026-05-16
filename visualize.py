import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


from SANeuTF import SANeuTF 
from data import load


def get_gates(model, a_id, b_id, c_ids):

    model.eval()
    device = next(model.parameters()).device
    
    # 构造 Tensor
    a_tensor = torch.tensor([a_id] * len(c_ids)).to(device)
    b_tensor = torch.tensor([b_id] * len(c_ids)).to(device)
    c_tensor = torch.tensor(c_ids).to(device)
    
    with torch.no_grad():
        # 触发前向传播，让模型把 Gate 存入 saved_gates
        _ = model(a_tensor, b_tensor, c_tensor)
        # 取出保存的 Gate，维度为 [len(c_ids), 125]
        gates = model.st_gating.saved_gates 
        
    return gates


def plot_temporal_periodicity(model, a_id=1, b_id=1, target_dim=80): 
    print(f"正在绘制地点(A={a_id}, B={b_id})，维度 {target_dim} 的时间周期折线图...")
    
    months = list(range(60)) 
    
    # 提取特定地点、特定维度的 Gate 数据
    gates_over_time = get_gates(model, a_id=a_id, b_id=b_id, c_ids=months)
    gate_values = gates_over_time[:, target_dim]
    
    plt.figure(figsize=(10, 4))
    plt.plot(months, gate_values, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=5)
    
    # 拼接标题
    dynamic_title = f"Temporal Periodicity at Location (A={a_id}, B={b_id}) - Channel {target_dim}"
    plt.title(dynamic_title, fontsize=14, fontweight='bold')
    
    plt.xlabel("Time (Months)", fontsize=12)
    plt.ylabel("Gate Activation Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(f"temporal_periodicity_A{a_id}_B{b_id}_Dim{target_dim}.png", dpi=300)
    plt.show()

def plot_spatial_heterogeneity(model, target_time=6, loc1_a=20, loc1_b=20, loc2_a=10, loc2_b=10):
    print(f"正在绘制 时间(T={target_time})，地点1(A={loc1_a}, B={loc1_b}) 与 地点2(A={loc2_a}, B={loc2_b}) 的空间异质性热力图...")
    
    # 获取地点1 和 地点2 在同一时间的 125 维 Gate
    gate_loc1 = get_gates(model, a_id=loc1_a, b_id=loc1_b, c_ids=[target_time])[0]
    gate_loc2 = get_gates(model, a_id=loc2_a, b_id=loc2_b, c_ids=[target_time])[0]
    
    # Reshape 成 5 x 25 的矩阵
    matrix_loc1 = gate_loc1.reshape(5, 25)
    matrix_loc2 = gate_loc2.reshape(5, 25)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    
    # 画地点 1 的热力图
    sns.heatmap(matrix_loc1, ax=axes[0], cmap="YlOrRd", vmin=0, vmax=1, cbar=True)
    title_loc1 = f"Gate Activation Map - Loc 1 (A={loc1_a}, B={loc1_b}) @ T={target_time}"
    axes[0].set_title(title_loc1, fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Q × R (25)", fontsize=12)
    axes[0].set_ylabel("P (5)", fontsize=12)
    
    # 画地点 2 的热力图
    sns.heatmap(matrix_loc2, ax=axes[1], cmap="YlOrRd", vmin=0, vmax=1, cbar=True)
    title_loc2 = f"Gate Activation Map - Loc 2 (A={loc2_a}, B={loc2_b}) @ T={target_time}"
    axes[1].set_title(title_loc2, fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Q × R (25)", fontsize=12)
    axes[1].set_ylabel("P (5)", fontsize=12)
    
    plt.tight_layout()
    # 保存文件名
    filename = f"spatial_heterogeneity_T{target_time}_Loc1({loc1_a}-{loc1_b})_vs_Loc2({loc2_a}-{loc2_b}).png"
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_soft_modulation_histogram(model, num_samples=1000, target_dim=None):
    dim_info = f"Dim {target_dim}" if target_dim is not None else "All Dims"
    print(f"正在绘制软调制直方图 (采样数={num_samples}, 提取特征={dim_info})...")
    
    # 随机取 N 个不同的时间切片和地点组合
    a_ids = np.random.randint(0, model.max_aID, num_samples)
    b_ids = np.random.randint(0, model.max_bID, num_samples)
    c_ids = np.random.randint(0, model.max_cID, num_samples)
    
    model.eval()
    device = next(model.parameters()).device
    gates_list =[]
    
    with torch.no_grad():
        for a, b, c in zip(a_ids, b_ids, c_ids):
            _ = model(torch.tensor([a]).to(device), 
                      torch.tensor([b]).to(device), 
                      torch.tensor([c]).to(device))
            # 注意：这里不再直接 flatten，而是保留 125 的完整结构
            gates_list.append(model.st_gating.saved_gates[0]) 
            
    # 变成一个巨大的矩阵[num_samples, 125]
    all_gates_matrix = np.array(gates_list) 
    
    # 指定维度
    if target_dim is not None:
        plot_data = all_gates_matrix[:, target_dim]
        title_suffix = f"Channel {target_dim}"
        file_suffix = f"Dim{target_dim}"
    else:
        plot_data = all_gates_matrix.flatten()
        title_suffix = "All Channels"
        file_suffix = "All"
    
    plt.figure(figsize=(8, 5))
    plt.hist(plot_data, bins=50, color='#ff7f0e', edgecolor='black', alpha=0.7)
    
    # 打印出样本数量和指定的维度
    dynamic_title = f"Dynamic Gate Distribution ({num_samples} Samples) - {title_suffix}"
    # plt.title(dynamic_title, fontsize=14, fontweight='bold')
    plt.xlabel("Gate Value (0.0 to 1.0)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xlim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 保存文件名
    filename = f"soft_modulation_hist_{num_samples}samples_{file_suffix}.png"
    # plt.savefig(filename, dpi=300)
    plt.savefig(f'{filename}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    MODEL_PATH = "/workspaces/01_project/Results/PCTemp/PCTemp/d10.0/saneutf/Ranks5_5_5/models/best_mae.model"  
    

    DATA_PATH = "/workspaces/01_project/SNeuTucF/NeuTucF/dataset/PCTemp_p10.0_tr10.0.csv"
    train_dataset = load(DATA_PATH)
    max_A_index = torch.max(train_dataset.A_Indices).item() + 1
    max_B_index = torch.max(train_dataset.B_Indices).item() + 1
    max_C_index = torch.max(train_dataset.C_Indices).item() + 1
    print(f"数据维度: A={max_A_index}, B={max_B_index}, C={max_C_index}")

    # 构造相同的 Config
    config = {
        "dimensions":[max_A_index, max_B_index, max_C_index],
        "embedding_dims": [5, 5, 5],
        "flag_bias": False,
        "use_mps": False,
    }

    # 初始化模型
    print("正在初始化模型结构...")
    model = SANeuTF(config)
    
    # 将模型放到对应的设备上
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    # 加载训练好的权重
    print(f"正在加载模型权重: {MODEL_PATH}")
    model.load_model(MODEL_PATH)
    print("模型加载成功！开始绘图...\n")

    # 设定画图风格
    sns.set_theme(style="whitegrid")

    # 执行绘图
    plot_temporal_periodicity(model)
    plot_spatial_heterogeneity(model)
    plot_soft_modulation_histogram(model)
    
    print("绘图完成！图片已保存在当前目录下。")