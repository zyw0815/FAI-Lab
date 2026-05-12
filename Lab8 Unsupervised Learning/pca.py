import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. 生成示例数据（二维椭圆状数据，便于可视化）
np.random.seed(42)
# 创建具有明显主成分方向的合成数据
X = np.dot(np.random.rand(2, 2), np.random.randn(2, 100)).T

# 2. 数据标准化（PCA对尺度敏感，必须标准化）
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 3. 计算协方差矩阵（关键步骤：找到变量间关系）
cov_matrix = np.cov(X_std.T)  # 使用转置得到变量间的协方差

# 4. 特征值分解（获取主成分方向）
eigen_vals, eigen_vecs = np.linalg.eigh(cov_matrix)
# 调整排序（从大到小）和对应特征向量
sorted_index = np.argsort(eigen_vals)[::-1]
eigen_vals = eigen_vals[sorted_index]
eigen_vecs = eigen_vecs[:, sorted_index]

# 5. 选择主成分（这里选第一个主成分）
n_components = 1
principal_components = eigen_vecs[:, :n_components]

# 6. 数据投影（降维到一维）
X_pca_manual = X_std.dot(principal_components)

# 使用sklearn验证结果
pca = PCA(n_components=1)
X_pca_sklearn = pca.fit_transform(X_std)

# 可视化结果
plt.figure(figsize=(12, 5))

# 原始数据可视化
plt.subplot(1, 2, 1)
plt.scatter(X_std[:, 0], X_std[:, 1], alpha=0.7)
# 绘制特征向量方向
for vec, color in zip(eigen_vecs.T, ['r', 'b']):
    plt.arrow(0, 0, vec[0]*3, vec[1]*3, 
              color=color, width=0.02, head_width=0.2)
    plt.text(vec[0]*3.5, vec[1]*3.5, f'PC{color}', color=color)
plt.title('Original Data with Principal Components')
plt.xlabel('Feature 1'), plt.ylabel('Feature 2')
plt.grid(True)

# 降维结果可视化（一维数据用散点图展示）
plt.subplot(1, 2, 2)
plt.scatter(X_pca_manual, np.zeros(len(X_pca_manual)), alpha=0.7)
plt.title('Data After PCA Projection')
plt.xlabel('Principal Component 1'), plt.yticks([])
plt.grid(True)

plt.tight_layout()
plt.show()

# 输出关键参数
print("手动实现主成分方向：\n", principal_components)
print("sklearn主成分方向：\n", pca.components_.T)
print("主成分解释方差比例：", pca.explained_variance_ratio_)