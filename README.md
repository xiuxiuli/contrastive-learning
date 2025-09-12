🧭 Contrastive Learning Project

📌 项目概述

本项目实现并比较多种 对比学习 (Contrastive Learning) 方法，包括 SimCLR, MoCo, BYOL，并在 STL-10 / ImageNet-100 数据集上进行自监督表征学习。通过在下游任务（分类、检索、半监督学习）中的迁移实验，展示对比学习在企业应用中的潜力。

🔧 技术栈

语言：Python
框架：PyTorch (+ Lightning 可选)
工具：torchvision, scikit-learn, Faiss, Matplotlib/Seaborn

模块	            说明
Encoder	            ResNet-18 / ResNet-50
Projection Head	    2-layer MLP
Loss	            NT-Xent (SimCLR), InfoNCE (MoCo), BYOL loss
Methods	            SimCLR, MoCo, BYOL
Augmentations	    RandomResizedCrop, ColorJitter, GaussianBlur, HorizontalFlip

📂 数据集与实验路线
阶段 1：STL-10 → 主实验（无标签预训练 + 下游评估）
阶段 3：ImageNet-100 → 扩展实验（资源允许）

🧪 下游任务
Linear Probe：冻结 encoder，训练线性分类器
Fine-tuning：微调 encoder
Retrieval Demo：embedding 最近邻检索 (Faiss)
Semi-Supervised：无标签预训练 + 少量标签微调

🎓 涉及知识点
CNN (ResNet)
自监督学习 (Contrastive Learning)
损失函数 (InfoNCE, NT-Xent, BYOL loss)
负样本策略 (batch 内 vs memory queue)
数据增强与表示学习
下游任务迁移 (分类/检索/半监督)

🚀 可扩展创新点
方法对比 (SimCLR vs MoCo vs BYOL)
数据增强实验 (不同组合的影响)
Retrieval Demo（输入一张图返回相似图像）
mini-CLIP（图像 + 类别文本 embedding 对齐）

🏢 企业应用价值
图像检索/推荐：电商、医疗、制造业
半监督分类：低标注场景
多模态检索：图像 ↔ 文本（CLIP思路）
异常检测：金融风控、工业质检、安防

一句话：对比学习是“特征工厂”，能把无标签数据转化为企业可用表征。