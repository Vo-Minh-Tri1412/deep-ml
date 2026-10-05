import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    X = np.array(features)
    w = np.array(weights)
    y = np.array(labels)

    # 1. Tính kết quả thô z (sửa b thành bias)
    z = X @ w + bias 
    
    # 2. Tính Sigmoid (thêm dấu ngoặc và dấu trừ)
    probabilities = 1 / (1 + np.exp(-z))
    
    # 3. Tính Mean Squared Error
    mse = np.mean((probabilities - y) ** 2)
    
    # 4. Làm tròn 4 chữ số thập phân và đưa về kiểu dữ liệu gốc theo đề bài
    probabilities_rounded = np.round(probabilities, 4).tolist()
    mse_rounded = float(np.round(mse, 4))
    
    return probabilities_rounded, mse_rounded
