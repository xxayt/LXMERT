[toc]



## LXMERT

论文：[LXMERT: Learning Cross-Modality Encoder Representations from Transformers (EMNLP2019)](https://arxiv.org/abs/1908.07490)

代码：[airsplay/lxmert (github.com)](https://github.com/airsplay/lxmert)

**选择原因**：结构简单（可改进的空间大）；[效果较好](https://eval.ai/web/challenges/challenge-page/163/leaderboard/498)（VQA 2.0挑战赛中本结构的Ensemble版本至今第3名，github version版本至今第9名）；数据集（VQA 2.0）测试在可接受范围

### 1 结构

<img src=".\image\LXMERT结构.png" alt="结构" style="zoom:60%;" />

- object-level image embeddings 图像嵌入层：RoI feature + position feature
- word-level sentence embeddings 文本嵌入层：word embedding + index embedding
- Object-Relationship Encoder：（$N_R$ 层）self-attention + FFN
- Language Encoder：（$N_L$ 层）self-attention + FFN
- Cross-Modality Encoder：（$N_X$ 层）cross-attention + self-attention + FFN

### 2 预训练

<img src=".\image\LXMERT预训练任务.png" alt="预训练任务" style="zoom:60%;" />

- pre-train tasks：5个
  - 针对文本：Masked Cross-Modality LM
  - 针对图像：Masked Object Prediction via RoI-feature regression / via detected-label classification
  - 针对跨模态：Cross-Modality Matching，Image Question Answering (QA)
- 输入：图像（物体特征）和相关句子（例如，标题或问题）

### 3 微调



#### VQA

- 数据集：VQA 2.0
  - 图像：使用对MS COCO数据集，用faster-rcnn提取目标特征的图像（训练集17G，测试集8G），其中已包含VQA 2.0数据集的图像
  - 文本：VQA 2.0（训练集125M，测试集61M）
  
- 训练：
  - 超参：$4 \text{ epoch},32 \text{ batch size}, 5e-5 \text{lr}$ 
  - 我的显卡：2080Ti，显存11264MiB
  - 训练耗时：**80min/epoch**
  - 验证耗时：**共约60min**
  
- VQA效果（上传[VQA：视觉问答 (visualqa.org)](https://visualqa.org/challenge.html)提交验证）

  |                  | src repo result | my result |
  | :--------------: | :-------------: | :-------: |
  | Local Validation |      69.9%      |  69.507%  |
  |     Test-Dev     |     72.42%      |           |
  |  Test-Standard   |     72.54%      |           |

#### GQA

- 数据集
  - 图像：分别对Visual Genome、MS COCO数据集，用faster-rcnn提取训练和测试的目标特征图像（训练集30G，测试集。。），
  - 文本：
- 训练：
  - 超参：$\text{epoch: }4,\text{batch size: 32},\text{lr: }1e-5$ 
  - 我的显卡：2080Ti，显存11264MiB

#### NLVR



## MCAN

[Deep Modular Co-Attention Networks for Visual Question Answering (CVPR2019)](https://arxiv.org/abs/1906.10770)

### 1 结构

<img src=".\image\MACN结构.png" alt="MACN结构" style="zoom:50%;" />

- 问题和图像的表示（Question and Image Representations）
- 深度协同注意力学习（Deep Co-Attention Learning）
- 多模态融合和分类输出（Multimodal Fusion and Output Classifier）

### 2 训练



## ViLBERT

[ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks (NIPS2019)](https://arxiv.org/abs/1908.02265)

