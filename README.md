# EE427 Video Signal Processing

## 1.Content

1.Reference answers to final examination questions

2.PPT Example reference answer



【Swin Transformer Paper Intensive reading 【 Paper intensive reading 】

https://www.bilibili.com/video/BV13L4y1475U

Don't understand swin can see Mu God talk about this



## 2.Find out a mistake

As there is no standard answer, if you detect any error in the answer, it is due to my limited knowledge level. 

Please leave a message or send an email or some Wechat messages to me and I will update the version as soon as possible. 

My Email: freshmanfreshman0011@gmail.com

My Wechat: freshman2233

## 4.GitHub Sponsors

If you like this project, please consider supporting us through GitHub Sponsors! Thank you for your support!



## 5. Exam scope: All

1. Introduction and foundation

2. Image compression & JPEG standard

3. Video compression and standards

4. Artificial Intelligence model and architecture

5. Video analysis and understanding

6. Basic models and generative artificial intelligence

## 6. Details

1.Introduction and foundation

1.1 Image and video fundamentals

1.2. Video signal processing and application

1.3. Image and video standards

1.4. Basic video analysis and understanding

1.5. Emerging themes



2.Image compression & JPEG standard

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





3.Video compression and standards

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





4.Artificial Intelligence model and architecture

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





4.2 Recurrent Neural Networks (RNN) and Long Short-term Memory (LSTM)

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





5.Video analysis and understanding

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









6.Basic models and generative artificial intelligence

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
