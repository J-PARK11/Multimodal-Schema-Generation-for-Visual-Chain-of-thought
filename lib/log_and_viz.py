import os
# import cv2
import torch
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import MinMaxScaler


# Seaborn 스타일 설정
sns.set(style="whitegrid")
colors = sns.color_palette("husl", 40)

def plot_loss(train_loss, valid_loss, epochs, len_step_per_epoch, save_path=None):

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    
    # 학습 로스
    train_xticks = list(map(int,list(train_loss.keys())))
    train_xvalues = list(train_loss.values())
    plt.plot(train_xticks, train_xvalues, 'b-', label='Training Loss', linewidth=2)
    
    # 검증 로스
    valid_xticks = list(map(int,list(valid_loss.keys())))
    valid_xvalues = list(valid_loss.values())
    plt.plot(valid_xticks, valid_xvalues, 'r-', marker='x', label='Validation Loss', linewidth=2)

    # 그래프 제목과 레이블 설정
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # 에포크 별 수렴 추이
    # Rectangle((x좌표, y좌표), 가로길이, 세로길이)
    max_value = max(train_xvalues+valid_xvalues)
    min_value = min(train_xvalues+valid_xvalues)
    for i in range(epochs):
        rect = patches.Rectangle((len_step_per_epoch*i, min_value), len_step_per_epoch, max_value, linewidth=0, edgecolor='none', facecolor=colors[i*2], alpha=0.1)
        plt.gca().add_patch(rect)

    # 범례 및 기타 설정
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # 저장 경로가 있으면 파일로 저장
    plt.savefig(save_path, format='png', dpi=300)

def plot_lr_loss(lr_log, epochs, len_step_per_epoch, save_path=None):

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    
    # LR 그래프
    lr_xticks = list(map(int,list(lr_log.keys())))
    lr_xvalues = list(lr_log.values())
    plt.plot(lr_xticks, lr_xvalues, 'b-', marker='o', label='LR', linewidth=2)

    # 그래프 제목과 레이블 설정
    plt.title('LR Scheduler Updated Graph', fontsize=14, fontweight='bold')
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('LR', fontsize=12)
    
    # 에포크 별 수렴 추이
    # Rectangle((x좌표, y좌표), 가로길이, 세로길이)
    max_value = max(lr_xvalues)
    for i in range(epochs):
        rect = patches.Rectangle((len_step_per_epoch*i,0), len_step_per_epoch, max_value, linewidth=0, edgecolor='none', facecolor=colors[i*9], alpha=0.1)
        plt.gca().add_patch(rect)

    # 범례 및 기타 설정
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # 저장 경로가 있으면 파일로 저장
    plt.savefig(save_path, format='png', dpi=300)

def visualize_attention_map(args, attention_map, dcp_text, code_text, head, puzzle_name):
    """
    KxT 형태의 어텐션 맵을 시각화하는 함수.
    
    Parameters:
    - attention_map (torch.Tensor or np.ndarray): K x T 형태의 어텐션 맵. (ex: torch.Tensor or numpy array)
    - title (str): 그래프의 제목. 기본값은 "Attention Map".
    """
    
    att_map_save_path = os.path.join(args.save_root, f'{args.load_ckpt_path.split(".")[0]}_{puzzle_name}_3L_{head}H_attention_map.png')
    title = f'{puzzle_name}_3L_{head}H_attention_map'
    
    # 텐서를 NumPy 배열로 변환
    try:
        if isinstance(attention_map, torch.Tensor):
            attention_map = attention_map.type(torch.float32)
            scaler = MinMaxScaler()
            attention_map = attention_map.detach().cpu().numpy()
            attention_map = scaler.fit_transform(attention_map)
        
        # 그래프 크기 설정
        if attention_map.shape[0] > attention_map.shape[1]:
            Height, Width = 40, 30
        else:
            Height, Width = 30, 40
        plt.figure(figsize=(Width, Height))
        
        # 어텐션 맵 시각화
        plt.imshow(attention_map, cmap='viridis', aspect='auto')
        plt.colorbar(label='Cross Attention Weight')
        plt.title(title)
        plt.xlabel("Code Text")
        plt.ylabel("Dcp Text")
        
        # 눈금 라벨을 추가하여 시각적으로 구분 가능하게 설정
        plt.xticks(ticks=np.arange(attention_map.shape[1]), labels=code_text, rotation=30)
        plt.yticks(ticks=np.arange(attention_map.shape[0]), labels=dcp_text)
        
        plt.savefig(att_map_save_path)
        print(puzzle_name, attention_map.shape, att_map_save_path)
    except:
        pass