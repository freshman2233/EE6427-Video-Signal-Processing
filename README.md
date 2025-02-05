# EE6427 Video Signal Processing

# 1.Introduction

An advanced course bridging the fundamentals of **video encoding** with cutting-edge **artificial intelligence** and **computer vision** techniques. 

In the first segment, we delve into the core concepts of image and video compression—covering standards like **DCT, JPEG, MPEG, and H.26x**—to understand how visual data is efficiently coded. 

We then transition into modern AI methods, discussing **convolutional neural networks (CNNs)**, **recurrent neural networks (RNNs)**, **transformers**, and the latest trends in computer vision such as **object detection**, **multi-target tracking**, and **pose estimation**, and the **foundation models, and large language models (LLMs)**.

This repository serves as a comprehensive resource for students and enthusiasts alike. 

1. **Personal Solutions to Past Exams** – Detailed, step-by-step write-ups of previously tested questions to guide your revision and deepen conceptual understanding.
2. **PPT Example References** – Walkthroughs of example problems and exercises presented in the lecture slides, clarifying key ideas and methodologies.
3. **Analysis of Challenging Topics** – In-depth discussions and breakdowns of complex areas in both video compression and AI, helping you navigate common pitfalls and master advanced concepts.

By consolidating these resources, the repository offers a practical guide to help you master video encoding fundamentals, tackle challenging AI and computer vision problems, and explore emerging breakthroughs in foundation models and LLMs.

# 2.**Course Aims**

The objective of this course is to provide you with knowledge in image and video signal processing. 

This course focuses on advanced topics in image and video processing, especially on the image filter, image and video compression, and some international standards for image and video processing. 

All of these topics are important to the understanding of image and video technologies and applications. 

This course will also arouse your interest in image and video processing topics and further motivate you towards developing your career in the area of image and video processing.



# 3.**Reading and References**

Textbook:

(1) Yun Q. Shi and Huifang Sun, Image and Video Compression for Multimedia Engineering: Fundamentals, Algorithms, and Standards, CRC Press, 3rd Edition, 2019. 

(2) Y. Wang, J. Ostermann, and Y.-Q. Zhang, Video Processing and Communications, Prentice Hall, 2002.



References: 

(1) Iain E.G. Richardson, H.264 and MPEG-4 Video Compression. Video Coding for Next-generation Multimedia, John Wiley & Sons, 2004. 

(2) John W. Woods, Multidimensional Signal, Image, and Video Processing and Coding, Academic Press, 2011.



Video:

**Swin Transformer** Paper Intensive reading https://www.bilibili.com/video/BV13L4y1475U



# 4.List of  Github

```
EE427 Video Signal Processing
├─1.Exam
│      21-S1-Q1-v4.pdf
│      21-S1-Q2-v2.pdf
│      21-S1-Q3-v5.pdf
│      21-S1-Q4-NO.pdf
│      21-S1-Q5-NO.pdf
│      22-S1-Q1-v3.pdf
│      22-S1-Q2-v2.pdf
│      22-S1-Q3-v2.pdf
│      22-S1-Q4-v3.pdf
│      22-S1-Q5-NO.pdf
│      23-S1-Q1-v3.pdf
│      23-S1-Q2-v2.pdf
│      23-S1-Q3-v2.pdf
│      23-S1-Q4-v3.pdf
│      23-S2-Q1-v4.pdf
│      23-S2-Q2-v2.pdf
│      23-S2-Q3-v3.pdf
│      23-S2-Q4-v2.pdf
│      
├─2.PPT Example
│      Example1.1.3.3.pdf
│      Lect2-Excercise-2D-DCT Using Matrix Implementation-v2.pdf
│      Lect2-Excercise-JPEG解码器.pdf
│      Lect2-Excercise-Traditional 2D-DCT.pdf
│      Lect2-Excercise-基函数与量化.pdf
│      Lect2-Excercise-霍夫曼编码 1 .pdf
│      Lect2-Excercise-霍夫曼编码 2-v2.pdf
│      Lect3-Excercise-H264-intro code.pdf
│      Lect3-Excercise-H264-运动补偿-v2.pdf
│      Lect3-Excercise-H264-运动补偿.pdf
│      Lect3-Excercise-MPEG-1.pdf
│      Lect3-Excercise-MPEG-2.pdf
│      Lect3-H.264 Example of Integer Transform.pdf
│      Lect4-Example-Softmax Loss.pdf
│      Lect4-Excercise-CNN-FC-Softmax.pdf
│      Lect4-Excercise-CNN.pdf
│      Lect4-Excercise-LSTM-v2.pdf
│      Lect4-Excercise-RNN-v2.pdf
│      Lect5-Exercises-MOT.pdf
│      Lect5-Exercises-Object Detection (RCNN vs YOLO).pdf
│      Lect5-Exercises-Temporal Shift Module (TSM)-v2.pdf
│      
├─3.Understand
│      1.Swin Transformer：使用移位窗口的分层视觉转换器.pdf
│      10.MPEG-1 P frame diagram.pdf
│      11.LSTM&RNN formula.pdf
│      12.Transformer Architecture.pdf
│      13.JPEG encoder.pdf
│      14.IQ IDCT理解.pdf
│      15.卡西欧fx-991cnx，三角函数，角度转弧度.pdf
│      16.EE6427计算题整理-山芋爱辣椒.pdf
│      17.平滑图像和有纹理的图像区别.pdf
│      18.Fms & Examples.pdf
│      19.different-pre-trained&fine-tuning.pdf
│      2.Understand Swin Transform5.1.3.2.pdf
│      20.FMs challenges.pdf
│      21.Adapt methods.pdf
│      22.LLM.pdf
│      23.注意力计算PPT例子.pdf
│      24.MLLMs.pdf
│      25.Different Swin Transformer & ViT.pdf
│      26.padding.pdf
│      3.性能指标metric.pdf
│      4.流程图总结.pdf
│      5.JPEG-量化Q.pdf
│      6.JPEG decoder.pdf
│      7.模型比较.xlsx
│      8.VGG.pdf
│      9.正交矩阵.pdf
│      EE6427-热门考点分析.xlsx
│      考试必背.pdf
│      
└─4.Resource
    │  EE6427_28082022.pdf
    │  
    └─1.信息论与编码
        ├─1.国防科技大学-信息论与编码基础
        │      信息论与编码.docx
        │      
        └─2.信息论与编码原理速成
                信息论与编码原理不挂科-1-信息论简介与概率论复习.pdf
                信息论与编码原理不挂科-2-离散信源及其信息测度.pdf
                信息论与编码原理不挂科-3-离散信道及信道容量.pdf
                信息论与编码原理不挂科-4-波形信源与波形信道.pdf
                信息论与编码原理不挂科-5-无失真信源编码.pdf
                信息论与编码原理不挂科-6-限失真信源编码简介.pdf
                信息论与编码原理不挂科-7-有噪信道编码.pdf        
```



# 5. Exam scope: All

1. Introduction and foundation
2. Image compression & JPEG standard
3. Video compression and standards
4. Artificial Intelligence model and architecture
5. Video analysis and understanding
6. Basic models and generative artificial intelligence

1.介绍与基础

2.图像压缩& JPEG标准

3.视频压缩和标准

4.人工智能模型和体系结构

5.视频分析与理解

6.基础模型与生成式人工智能

# 6. Content

## 6.1 English

### 1.Introduction and foundation

1.1 Image and video fundamentals

1.2. Video signal processing and application

1.3. Image and video standards

1.4. Basic video analysis and understanding

1.5. Emerging themes



### 2.Image compression & JPEG standard

2.1. Terms and concepts

2.1.1. What is compression/encoding?

2.1.2. Compression ratio

2.1.3. Fundamentals of information theory



2.2. Entropy coding

2.2.1. Huffman coding



2.3. Image and video compression

2.3.1. Why compress images/videos?

2.3.2. Why can it be compressed?

2.3.3 Space Redundancy

2.3.4 Time redundancy

2.3.5 Coding redundancy

2.3.6 Psychovisual redundancy

2.3.7 Lossy and lossless compression

2.3.8 Distortion measures/measurements



2.4. Transform based coding/compression

2.4.1 Transform coding

2.4.2 Image compression based on transformation

2.4.3 Conversion

2.4.4 Quantization

2.4.5 Scalar quantization

2.4.6 Coding





2.5. Discrete Cosine Transform (DCT)

2.5.1 Discrete Cosine Transform (DCT)

2.5.2 Why DCT?

2.5.3 Transformation and basis function

2.5.4 Traditional DCT transform

2.5.5 Matrix DCT transformation





2.6.JPEG standard

2.6.1 Observation

2.6.2 JPEG

2.6.3 Main stages of baseline JPEG

2.6.4 JPEG encoder

2.6.5 Image segmentation

2.6.6 Image Preparation



2.6.7 Forward DCT

2.6.7.1 Why 2D-DCT?

2.6.7.2JPEG DCT base function

2.6.7.3 Changes of Bases in DCT

2.6.7.4DCT coefficient



2.6.8 Quantization

2.6.8.1 Quantization table

2.6.8.2 Quantization

2.6.8.3 Quantify the effect



2.6.9 Entropy coding

2.6.9.1 Zigzag Scan

2.6.9.2 Differential coding of DC coefficient

2.6.9.3 Run Code RLE

2.6.9.4 Huffman coding



2.6.10 Image quality vs DCT components



2.6.11 Framework construction

2.6.11.1JPEG Bitstream format



2.6.12 JPEG decoder





### 3.Video compression and standards

3.1. Video coding and standards

3.1.1 Video structure

3.1.2 Macro module

3.1.3 Main ideas of video compression

3.1.3.1 Motion estimation and compensation

3.1.4 Intra-frame and inter-frame compression

3.1.5 Brightness & chroma

3.1.6 Chrominance secondary sampling





3.2.MPEG standard

3.2.1MPEG Basics

3.2.1.1 Introduction

3.2.1.2MPEG Video Structure

3.2.1.3 Picture Group (GOPs)

3.2.1.4i Frame, p Frame, and b Frame

3.2.1.5MPEG-1 Display and encoding sequence

3.2.1.6 GOP problem of picture group



3.2.2 Motion estimation and compensation

3.2.2.1 Motion estimation

3.2.2.2 Motion compensation

3.2.2.2.1 Mean Absolute Difference (MAD)



3.2.2.3 Motion estimation method

3.2.2.3.1 Complete search

3.2.2.3.2 Three-Step Search

3.2.2.3.3 Logarithmic Search

3.2.2.3.4 Layered Search



3.2.3 History and Milestones

3.2.3.1 Video Coding Standardization Organization

3.2.3.2 Key coding standards and Milestones

3.2.3.3MPEG Overview

3.2.3.4H.26x Summary

3.2.3.5 Other video coding Standards



3.2.4 mpeg-1

3.2.4.1. Overview

3.2.4.2 I, P & B frame coding

3.2.4.2.1 I-frame Coding Flow Chart

3.2.4.2.2 P-frame Coding Flow Chart

3.2.4.2.3 B-frame Coding Flow Chart

3.2.4.2.4 Frame Size

3.2.4.3. Bit Stream





3.2.5 mpeg-2

3.2.5.1. Overview

3.2.5.2. Scalability

3.2.5.3. Types of extensibility

3.2.5.3.1. The SNR is scalable

3.2.5.3.2. Space Scalability

3.2.5.3.3. Time Scalability

3.2.5.3.4. Mixed Scalability

3.2.5.3.5 Data Partition



3.2.6 mpeg-4

3.2.6.1 Overview

3.2.6.2 Object-based Video Coding vs Frame-based video Coding

3.2.6.3 Video Object Plane VOP Coding





3.2.7 the MPEG - 7







3.3.H.26x standard

3.3.1 H.261

3.3.1.1 Overview

3.3.1.2 I Frame coding

3.3.1.3 P-frame coding





3.3.2 H.262

3.3.3 H.263 / H.263+

3.3.4 H.264

3.3.4.1 H264 Encoder

3.3.4.2 Motion compensation

Other options in 3.3.4.3GOP

3.3.4.4 Integer Conversion

3.3.4.5 Internal Coding

3.3.4.6 Intra-Loop Block Filtering

3.3.4.7 Context-Adaptive Binary Arithmetic Coding (CABAC)

3.3.4.8 Bit Stream Structure

3.3.4.9 Scalable Video Coding (SVC)

3.3.4.10 Multi-View Video Coding (MVC)





3.3.5 H.265, H.266

3.3.5.1H.265: Introduction

3.3.5.2 H.265: Improvements to H.264

3.3.5.3H.265 :Intra Coding

3.3.5.4 H.266: Overview



3.4. New/emerging codes and standards

3.4.1MPEG-I: Immersive media

3.4.2 Immersive Video (MIV)





### 4.Artificial Intelligence model and architecture

4.1 Convolutional Neural Networks (CNN)

4.1.1 Introduction

4.1.2 Linear classifier

4.1.2.1 Biological neural networks

4.1.2.2 Multi-Layer Perceptron (MLP)

4.1.2.3 Linear classifier

4.1.2.4 Loss function/Error function

4.1.2.4.1 Square loss

4.1.2.4.2 Mean square error MSE

4.1.2.4.3 Mean Absolute Error (MAE)

4.1.2.4.4 Softmax losses







4.1.3 Convolutional Neural Networks (CNN)

4.1.3.1 CNN Architecture

4.1.3.2 Convolution layer

4.1.3.3 Activating the Function Layer

4.1.3.4 Pooling Layer

4.1.3.5 Full Connection Layer

4.1.3.6 Probability Mapping Layer Softmax



4.1.4 CNN training and optimization

4.1.4.1 Random Gradient Descent (SGD)

4.1.4.2 Learning rate





4.1.5 Well-known CNN architecture





4.1.6 Applications





#4.2 Recurrent Neural Networks (RNN) and Long Short-term Memory (LSTM)

4.2.1 Introduction

4.2.2 Recurrent Neural Network (rnn)

4.2.2.1 Original RNN(Vanilla RNN)





4.2.3 RNN training and optimization

4.2.3.1 Calculation Diagram

4.2.3.2 Batch Training

4.2.3.3 Time Backpropagation (BPTT)

4.2.3.4 Truncation Time Backpropagation

4.2.3.5 Gradient Explosion gradient disappears



4.2.4 Long Short-term Memory (LSTM)

4.2.4.1 LSTM Architecture

4.2.4.2 RNN and LSTM





4.2.5 Applications





4.3 Transformer

4.3.1 Concept of attention

4.3.1.1 What is Attention?

4.3.1.2 How is Attention calculated?

4.3.1.3 Scaling dot product attention



4.3.2 Transformer Architecture

4.3.2.1 Attention is all you need

4.3.2.2 Transformer encoder

4.3.2.3 Transformer decoder

4.3.2.4 Transformer Structure

4.3.2.5 Overview of the Self-Focused Layer

4.3.2.6 Cross Attention/encoder-decoder self-attention

4.3.2.7 More details

4.3.2.8 Multiple attention

4.3.2.9 Location Code

4.3.2.10 Remaining Connection and Layer Specifications

4.3.2.11 Transformer encoder + decoder architecture

4.3.2.12 Final linear and Softmax layer

4.3.2.13 LSTM vs Transformer



4.3.3 ViT





### 5.Video analysis and understanding

5.1 Object Detection

5.1.1 Introduction

5.1.2 Object Detection Methods

5.1.2.1 Secondary Detector

5.1.2.2 Primary Detector

5.1.2.3 Lightweight detector







5.1.3 Emerging/New methods

5.1.3.1Swin Transform

5.1.3.2 Detecting the Transform (DETR)







5.2. Object tracking

5.2.1 Introduction



5.2.2 Multi-Target Tracking (MOT)

5.2.2.1 Tracing by Detection

5.2.2.2 Motion model: Kalman filter

5.2.2.3 Data Association

5.2.2.4 Deep SORT







5.2.3 New methods in MOT

5.2.3.1 Trackformer





5.3. Attitude estimation

5.3.1 Introduction

5.3.2 Individual Pose estimation

5.3.2.1 Regression Method

5.3.2.2 Body Part Detection. Body part detection



5.3.3 Multi-person pose estimation

5.3.3.1 Top-down: HRNet (High Resolution Network)

5.3.3.2 Bottom-up: OpenPose





5.3.4 Emerging/New methods

5.3.4.1 Estimation of TransPose based on attitude of transformer







5.4. Video/Human Motion Recognition (HAR)

5.4.1 Introduction

5.4.2 Two-Stream Networks Two-Stream networks

5.4.3 3D CNNs



5.4.4 Efficient video modeling

5.4.4.1 Temporal Shift Module (TSM)



5.4.5 Emerging/New methods

5.4.5.1 Video Swin Transforme for video rotary transformer

5.4.5.2 Skeleton-based network









### 6.Basic models and generative artificial intelligence

6.1 Basic Model (FMs)

6.1.1 Status quo of artificial intelligence

6.1.2 Basic Model Basics

6.1.3 Adjust or fine-tune the fm

6.1.3.1 What Is Fine Tuning?

6.1.3.2 Fine Tuning Process

6.1.3.3.1 Adapter Tuning

6.1.3.3.2 Low Rank Adapter (LoRA)

6.1.3.3.3 Quantifying Low-Rank Adapters (QLoRA)

6.1.3.3.4 Prefix Tuning

6.1.3.3.5 Prompt Optimization







6.2 Large Language Models (LLMs)

6.2.1 Introduction

6.2.2 Bidirectional Encoder Representation of transformer (BERT)

6.2.2.1 Stage 1: Pre-training

6.2.2.2 Phase 2: Fine-tuning for a specific task

6.2.3 Generating a Pre-Trained Transformer (GPT)

6.2.4 Well-known LLMS

6.2.5 Emerging Trends

6.2.5.1 Multimodal LLM (MLLM)



## 6.2 Chinese

### 1.介绍与基础

1.1图像和视频基础

1.2.视频信号处理与应用

1.3.图像和视频标准

1.4.视频分析与理解基本

1.5.新兴主题



### 2.图像压缩& JPEG标准

2.1.术语和概念

2.1.1.什么是压缩/编码？

2.1.2.压缩比

2.1.3.信息论基础 



2.2.熵编码

2.2.1.霍夫曼编码 



2.3.图像和视频压缩

2.3.1.为什么要压缩图像/视频? 

2.3.2.为什么可以压缩? 

2.3.3空间冗余 

2.3.4时间冗余 

2.3.5编码冗余 

2.3.6心理视觉冗余 

2.3.7有损和无损压缩 

2.3.8扭曲措施/量度 



2.4.基于变换的编码/压缩

2.4.1变换编码 

2.4.2基于变换的图像压缩 

2.4.3变换

2.4.4量化 

2.4.5标量量化 

2.4.6编码 





2.5.离散余弦变换(DCT)

2.5.1离散余弦变换(DCT) 

2.5.2为什么DCT ? 

2.5.3变换与基函数 

2.5.4传统DCT变换

2.5.5矩阵DCT变换





2.6.JPEG标准

2.6.1观察 

2.6.2JPEG 

2.6.3基线JPEG的主要阶段 

2.6.4 JPEG编码器 

2.6.5 图像分割 

2.6.6 图像准备 



2.6.7前向DCT 

2.6.7.1为什么是2D-DCT? 

2.6.7.2JPEG DCT基函数 

2.6.7.3DCT中基的变化

2.6.7.4DCT系数 



2.6.8量化 

2.6.8.1量化表 

2.6.8.2量化 

2.6.8.3量化效应 



2.6.9熵编码 

2.6.9.1锯齿形扫描 

2.6.9.2直流系数的差分编码 

2.6.9.3游程编码RLE

2.6.9.4霍夫曼编码 



2.6.10图像质量vs DCT组件 



2.6.11框架构建 

2.6.11.1JPEG比特流格式 



2.6.12 JPEG解码器 





### 3.视频压缩和标准

3.1.视频编码与标准

3.1.1视频结构 

3.1.2宏模块 

3.1.3视频压缩的主要思想 

3.1.3.1运动估计与补偿 

3.1.4帧内和帧间压缩 

3.1.5亮度&色度

3.1.6色度二次抽样 





3.2.MPEG标准

3.2.1MPEG基础

3.2.1.1介绍

3.2.1.2MPEG视频结构 

3.2.1.3图片组(GOPs) 

3.2.1.4i帧，p帧，b帧 

3.2.1.5MPEG-1显示和编码顺序 

3.2.1.6图片组的GOP问题 



3.2.2运动估计与补偿

3.2.2.1运动估计 

3.2.2.2运动补偿 

3.2.2.2.1平均绝对差(MAD)



3.2.2.3运动估计方法 

3.2.2.3.1完整的搜索 

3.2.2.3.2三步搜索

3.2.2.3.3对数搜索 

3.2.2.3.4分层搜索 



3.2.3历史与里程碑 

3.2.3.1视频编码标准化组织 

3.2.3.2关键编码标准和里程碑 

3.2.3.3MPEG概述 

3.2.3.4H.26x总结 

3.2.3.5其他视频编码标准 



3.2.4MPEG-1

3.2.4.1.概述 

3.2.4.2  I, P & B帧编码 

3.2.4.2.1 I帧编码流程图 

3.2.4.2.2 p帧编码流程图 

3.2.4.2.3 B帧编码流程图 

3.2.4.2.4 帧大小 

3.2.4.3.比特流 





3.2.5MPEG-2 

3.2.5.1.概述 

3.2.5.2.可伸缩性 

3.2.5.3.可扩展性的类型

3.2.5.3.1.信噪比可扩展

3.2.5.3.2.空间可扩展性 

3.2.5.3.3.时间可伸缩性 

3.2.5.3.4.混合可扩展性

3.2.5.3.5数据分区 



3.2.6MPEG-4

3.2.6.1概述

3.2.6.2基于对象的视频编码vs基于帧的视频编码 

3.2.6.3视频对象平面VOP编码





3.2.7MPEG-7







3.3.H.26x标准

3.3.1 H.261

3.3.1.1概述

3.3.1.2 I帧编码 

3.3.1.3 P帧编码 





3.3.2 H.262

3.3.3 H.263 / H.263+

3.3.4 H.264

3.3.4.1 H264编码器 

3.3.4.2运动补偿

3.3.4.3GOP中的其他选项 

3.3.4.4 整数变换 

3.3.4.5 内部编码 

3.3.4.6 循环内去块滤波 

3.3.4.7 上下文自适应二进制算术编码(CABAC) 

3.3.4.8 比特流结构 

3.3.4.9 可伸缩视频编码(SVC) 

3.3.4.10 多视图视频编码(MVC) 





3.3.5 H.265, H.266

3.3.5.1 H.265:介绍 

3.3.5.2 H.265:对H.264的改进 

3.3.5.3 H.265:Intra Coding

3.3.5.4 H.266:概述 



3.4.新的/新兴的编码和标准

3.4.1MPEG-I:沉浸式媒体 

3.4.2沉浸式视频(MIV) 





### 4.人工智能模型和体系结构

4.1卷积神经网络(CNN) 

4.1.1 介绍 

4.1.2 线性分类器 

4.1.2.1生物神经网络 

4.1.2.2 多层感知器(MLP) 

4.1.2.3 线性分类器 

4.1.2.4 损失函数/错误函数

4.1.2.4.1 平方损失 

4.1.2.4.2 均方误差MSE

4.1.2.4.3 平均绝对误差(MAE)

4.1.2.4.4 Softmax损失







4.1.3 卷积神经网络(CNN) 

4.1.3.1 CNN架构 

4.1.3.2 卷积层 

4.1.3.3 激活函数层 

4.1.3.4 池化层 

4.1.3.5 全连接层 

4.1.3.6 概率映射层 Softmax



4.1.4 CNN训练与优化 

4.1.4.1随机梯度下降(SGD)

4.1.4.2 学习率





4.1.5 众所周知的CNN架构 





4.1.6 应用





4.2循环神经网络(RNN)与长短期记忆(LSTM) 

4.2.1介绍 

4.2.2 递归神经网络(rnn)

4.2.2.1原版 RNN(香草RNN） 



 

4.2.3 RNN训练与优化 

4.2.3.1计算图 

4.2.3.2 批训练 

4.2.3.3 时间反向传播(BPTT) 

4.2.3.4 截断时间反向传播 

4.2.3.5梯度爆炸 梯度消失



4.2.4 长短期记忆(LSTM) 

4.2.4.1 LSTM架构 

4.2.4.2 RNN 与 LSTM 





4.2.5 应用





4.3 Transformer 

4.3.1 注意力概念 

4.3.1.1什么是注意力? 

4.3.1.2 注意力是如何计算的? 

4.3.1.3 缩放的点积注意力 



4.3.2 Transformer 架构

4.3.2.1 注意力就是你所需的一切 

4.3.2.2变压器编码器 

4.3.2.3 变压器解码器 

4.3.2.4 变压器结构 

4.3.2.5 自关注层概述 

4.3.2.6 交叉注意/编码器-解码器自我注意 

4.3.2.7 更多的细节 

4.3.2.8 多头注意力 

4.3.2.9 位置编码 

4.3.2.10 剩余连接和层规范 

4.3.2.11 变压器编码器+解码器架构 

4.3.2.12 最后的线性和Softmax层 

4.3.2.13 LSTM vs Transformer 



4.3.3 ViT





### 5.视频分析与理解

5.1对象检测 

5.1.1介绍 

5.1.2对象检测方法 

5.1.2.1二级检测器 

5.1.2.2一级检测器

5.1.2.3轻量级的探测器







5.1.3新兴/新方法

5.1.3.1Swin Transform

5.1.3.2检测Transform（DETR）







5.2.对象跟踪

5.2.1介绍



5.2.2多目标跟踪（MOT）

5.2.2.1通过检测追踪

5.2.2.2 运动模型：卡尔曼滤波

5.2.2.3 数据关联 

5.2.2.4 Deep SORT







5.2.3MOT中的新方法

5.2.3.1 Trackformer





5.3.姿态估计

5.3.1 介绍

5.3.2 单人姿态估计

5.3.2.1 回归方法Regression Method 

5.3.2.2 身体部位检测Body Part Detection



5.3.3多人姿态估计

5.3.3.1 自上而下：HRNet（高分辨率网络）

5.3.3.2 自下而上：OpenPose





5.3.4 新兴/新方法

5.3.4.1 基于变压器的姿态估计 TransPose







5.4.视频/人体动作识别（HAR）

5.4.1 介绍

5.4.2 双流网络 Two-Stream Networks

5.4.3 3D CNNs



5.4.4 高效视频建模

5.4.4.1时间偏移模块Temporal Shift Module (TSM)



5.4.5 新兴/新方法

5.4.5.1视频旋转变压器Video Swin Transforme

5.4.5.2 Skeleton-based网络









### 6.基础模型与生成式人工智能

6.1基础模型（FMs）

6.1.1人工智能的现状

6.1.2 基础模型的基础

6.1.3 调整/微调fm

6.1.3.1 什么是微调？

6.1.3.2 微调过程

6.1.3.3.1适配器调优

6.1.3.3.2 低秩适配器（LoRA）

6.1.3.3.3 量化低秩适配器（QLoRA）

6.1.3.3.4 前缀调优

6.1.3.3.5 提示优化







6.2大型语言模型（LLMs）

6.2.1 介绍

6.2.2 变压器的双向编码器表示（BERT）

6.2.2.1 阶段1：预训练

6.2.2.2 阶段2：针对特定任务进行微调

6.2.3 生成预训练变压器（GPT）

6.2.4 知名LLM

6.2.5新兴的趋势

6.2.5.1 Multimodal LLM (MLLM)





# 7.Disclaimer

All content in this  is based solely on the contributors' personal work, Internet data.
All tips are for reference only and are not guaranteed to be 100% correct.
If you have any questions, please submit an Issue or PR.
In addition, if it infringes your copyright, please contact us to delete it, thank you.



#### Copyright © School of Electrical & Electronic Engineering, Nanyang Technological University. All rights reserved.

