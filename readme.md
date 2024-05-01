# “智慧之眼”——智能医疗辅助诊断

## 1. 作品简介		

​		随着医学和技术的迅速发展，精准医疗已成为提高治疗效果和患者生存率的关键。智能医疗辅助诊断系统采用前沿的**计算机视觉、多模态**和**增强现实**技术，为临床医生提供高精度的**医学图像分析**和**立体可视化服务**，显著提升诊断的准确性和治疗的个性化水平。系统集成了**影像处理、二维分割、三维分割**和**三维重建**功能，旨在辅助医学专家进行更有效的疾病诊断和治疗规划。

​        本系统以多模态的方式，根据用户提供的**ROI区域**或**文本指令提示**，智能的分割出医学影像中的各种组织和器官，并对这些分割后的组织器官重建出**高度精确的立体影像**，再通过增强现实技术将3D影像进一步渲染到**现实世界**中。

​        这些先进技术不仅提高了**病变区域识别**的精确性和**手术视觉**的可靠性，还通过增强现实模拟显著**提升手术安全性和成功率**。从**初步诊断**到**临床手术执行**，本系统为整个医疗诊断和治疗过程提供了**全方位的精准辅助**，确保了治疗方案的高效实施。

![img](https://2024.jsjds.com.cn/Backend/Work/work/download-article-picture?name=20240263681713864594fiNa3C0GJqtU0S_mUgY3TmslGM1GDN.png)

## 2. 目录结构

```txt
--Checkpoints			模型检查点存放
--Models 			算法模型
--Processing			影像处理算法
--Reconstruction		三维重建算法
--Segmentation			影像分割算法
--tools				PyMed开发工具
--main.py			后端接口启动程序
--requirements.txt		安装需求说明
```

## 3. 安装说明

1.1 创建一个虚拟环境并激活

```
	conda create -n AIMed python=3.9
	conda activate AIMed
```

1.2 安装 [Pytorch2.0](https://pytorch.org/)﻿

1.3 安装相应的依赖包

```
	pip install -r requirements.txt
```

1.4 启动算法服务器

```
	uvicorn run:main --host 127.0.0.1 --port 8000 --reload
```
