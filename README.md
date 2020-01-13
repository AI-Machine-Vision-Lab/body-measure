# AR and Deep Learning based Automatic Human Body Measurement System
Submission for **Hacking the diaper** hackathon by Kimberly-Clark<br>
Team : Team Big Brain
## Problem statement
Based on a recent “fit” study, almost one-third of diaper users wear the wrong diaper size, while four out of ten mothers state that diaper fit is a significant concern at every stage of diapering. Despite disposable diapers being around for the last half-century, ensuring proper fit of diapers for babies can be confusing for parents. When babies wear the wrong diaper size, the chances of leakage and blowouts increase. Kimberly-Clark continues to partner with parents to not just provide a diaper that fits their baby perfectly, but also provide the technology and tools needed to reduce diaper fit challenges.<br>
## Objective
This project is aimed to provide a reliable solution for the parents to for the correct diaper size and predict change of diaper size upon future based on our AI/ML algorithm and AR technology.
## Solution:
This project uses a standalone model called PoseNet, for running real-time pose estimation in the browser using TensorFlow.js which can be run on a back-end web server.<br>
A lot of developers are using pretrained models because it is very easy to use and implement. Transfer learning is a huge use case for tensorflowjs. We can use pre-trained models and easily implement in tensorflow js.
Since we did not have image dataset of baby pictures, we've used general human body dataset.<br>
Though, it is very easy to train a model and implement in tensorflow.js.
### Using a pretrained model for classification (3 Steps)
#### Step 1 - Load Model
First, we'll need to import two files that define the structure of the model (model file) & its trained weights (weights manifest)<br>
![](https://camo.githubusercontent.com/097b5a83c332206b9803dc5cc6e15ec6cb2c28dd/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a6d5f51557a3736384d78414b344b586e76564c326e512e6a706567)
```
import * as tdc from '@tensorflow/tfjs-core';
import { loadFrozenModel } from '@tensorflow/tfjs-converter';
import { IMAGENET_CLASSES } from './imagenet_classes';
const MODEL_URL = '/models/mobilenet/optimized_model.pb';
const WEIGHTS_URL = '/models/mobilenet/weights_manifest.json';
const INPUT_NODE_NAME = 'input';
const OUTPUT_NODE_NAME = 'MobilenetV1/Predictions/Reshape_1';
const PREPROCESS_DIVIDOR = tfc.scalar(255 / 2);
export defualt class MobileNet {
async load () {
this.model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL);
```
#### Step 2 - Preprocessing
- A neural network has a specific input definition so we'll need to do some preprocessing in order to get your input into the right shape
- First, we'll convert pixels to TensorFlow.js input tensor
- Then, we'll crop the input if you want to use parts of the image
- Lastly, we'll set batch input dimensions to 0, since you only want to infer one image
```
preprocess(source) {
console.log('input size:' this.model.input.shape); // [224,224,3]
// memory enhancements - tells the system to throw away this tensor after usage
return tf.tidy(() => {
const input = tfc.fromPixels(source);
// crop the image to match the input size of mobilenet
// this is 224x224 px with 3 channel (RGB) color data
// get a square from the middle of the image
// resizing is done by the html5 canvas
const croppedImage = MobileNet.cropImage(input);
//mobilenet expects a batched input - build a [1, 224, 224, 3] tensor
const bachtedImage = croppedImage.expandDims(0);
//normalization of the pixel color channel values
//instead of 0-225 we get values between -1 and 1
return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
});
}
```
#### Step 3 - Inference
The inference of an input aka a class prediction is just two lines of code
```
predict(source) {
const processedInput = this.preprocess(source);
return this.model.predict(processedInput);
}
```
The result (prediction) will be a dictionary containing the probabilities for each class.
The max value within the dictionary is the most likely class
###### WHY TENSORFLOW.JS ?
We're using tensorflow.js for machine learning on web server.<br>
ML used to require lots of time and money to get started
Configurations, dependencies, hardware costs, lots of headaches
But now anyone can train and test ML models in the browser really easily using Tensorflow.js
Even python was more difficult (jupyter notebooks, numpy, scikit, pandas, etc)
ML in the browser means 
##### Privacy - Data is local, none leaves the clients device. Much safer. 
##### Wide distribution - JavaScript has one of the widest install bases of any language and framework. 
##### Distributed Computing - Leverage client side data from many users to help train a model
We've used Layer APIs for easier implementaion.<br>
![text](https://camo.githubusercontent.com/ee6d9e0bb3d04dde245b5cb662fe40d19c5b6541/68747470733a2f2f73332d61702d736f7574682d312e616d617a6f6e6177732e636f6d2f61762d626c6f672d6d656469612f77702d636f6e74656e742f75706c6f6164732f323031382f30342f3056344859625a74323850485a5a3361442e706e67)
### PoseNet
PoseNet can be used to estimate either a single pose or multiple poses
The single person pose detector is faster and simpler but requires only one subject present in the image
At a high level pose estimation happens in two phases
First, An input RGB image is fed through a convolutional neural network.
Either a single-pose or multi-pose decoding algorithm is used to decode poses, pose confidence scores, keypoint positions, and keypoint confidence scores from the model outputs
A Pose — at the highest level, PoseNet will return a pose object that contains a list of keypoints and an instance-level confidence score for each detected person. A Keypoint Position is 2D x and y coordinates in the original input image where a keypoint has been detected.
##### PoseNet currently detects 17 keypoints illustrated in the following diagram:
![keypoints](https://camo.githubusercontent.com/5a42c61a4e947dbc5f530b4e3b54f3ee97cc61a5/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a377144794c7049542d337334796c554c73726e7a38412e706e67)
![](https://camo.githubusercontent.com/36d3ddd2b7a162af115145d3fd6411020a570e32/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a7a5858775231366b707241574c50494f4b4372584c772e706e67)
![](https://camo.githubusercontent.com/01bc2d3caaf38e938686c9ac46392d94148e04e2/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a6d63616f76456f4c42745f416a306c7776312d7874412e706e67)
#### Pose confidence
A Pose confidence score determines the overall confidence in the estimation of a pose. It ranges between 0.0 and 1.0. It can be used to hide poses that are not deemed strong enough. <br>
AKeypoint Confidence Score determines the confidence that an estimated keypoint position is accurate. It ranges between 0.0 and 1.0. It can be used to hide keypoints that are not deemed strong enough. 
![confidence](https://camo.githubusercontent.com/7a01570050fd170459325f71aafdef79977cb01c/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a33626733434f31623479787167726a734761537742772e706e67)
## Demo

## References
1. https://github.com/CMU-Perceptual-Computing-Lab/openpose 
2. Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields: Zhe Cao Tomas Simon Shih-En Wei Yaser Sheikh: The Robotics Institute, Carnegie Mellon University<br>https://arxiv.org/pdf/1611.08050.pdf
3. https://www.youtube.com/watch?v=Nc8kZABv-KE
4. TFJS Docs https://js.tensorflow.org/tutorials/core-concepts.html
5. Shiffmans series https://www.youtube.com/watch?v=Qt3ZABW5lD0
