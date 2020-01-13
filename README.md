# AR and Deep Learning based Automatic Human Body Measurement System
Submission for **Hacking the diaper** hackathon by Kimberly-Clark<br>
Team : Team Big Brain
## Problem statement
Based on a recent “fit” study, almost one-third of diaper users wear the wrong diaper size, while four out of ten mothers state that diaper fit is a significant concern at every stage of diapering. Despite disposable diapers being around for the last half-century, ensuring proper fit of diapers for babies can be confusing for parents. When babies wear the wrong diaper size, the chances of leakage and blowouts increase. Kimberly-Clark continues to partner with parents to not just provide a diaper that fits their baby perfectly, but also provide the technology and tools needed to reduce diaper fit challenges.<br>
## Objective
This project is aimed to provide a reliable solution for the parents to for the correct diaper size and predict change of diaper size upon future based on our AI/ML algorithm and AR technology.
## Solution:
ML used to require lots of time and money to get started
Configurations, dependencies, hardware costs, lots of headaches
But now anyone can train and test ML models in the browser really easily using Tensorflow.js
Even python was more difficult (jupyter notebooks, numpy, scikit, pandas, etc)
ML in the browser means 
##### Privacy - Data is local, none leaves the clients device. Much safer. 
##### Wide distribution - JavaScript has one of the widest install bases of any language and framework. 
##### Distributed Computing - Leverage client side data from many users to help train a model
![text](https://camo.githubusercontent.com/ee6d9e0bb3d04dde245b5cb662fe40d19c5b6541/68747470733a2f2f73332d61702d736f7574682d312e616d617a6f6e6177732e636f6d2f61762d626c6f672d6d656469612f77702d636f6e74656e742f75706c6f6164732f323031382f30342f3056344859625a74323850485a5a3361442e706e67)
### PoseNet
PoseNet can be used to estimate either a single pose or multiple poses
The single person pose detector is faster and simpler but requires only one subject present in the image
At a high level pose estimation happens in two phases
First, An input RGB image is fed through a convolutional neural network.
Either a single-pose or multi-pose decoding algorithm is used to decode poses, pose confidence scores, keypoint positions, and keypoint confidence scores from the model outputs
A Pose — at the highest level, PoseNet will return a pose object that contains a list of keypoints and an instance-level confidence score for each detected person.

## references
1. https://github.com/CMU-Perceptual-Computing-Lab/openpose 
2. Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields: Zhe Cao Tomas Simon Shih-En Wei Yaser Sheikh: The Robotics Institute, Carnegie Mellon University<br>https://arxiv.org/pdf/1611.08050.pdf
