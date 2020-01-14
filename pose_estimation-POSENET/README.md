### Loading a pre-trained PoseNet Model

In the first step of pose estimation, an image is fed through a pre-trained model.  PoseNet **comes with a few different versions of the model,** corresponding to variances of MobileNet v1 architecture and ResNet50 architecture. To get started, a model must be loaded from a checkpoint:

```javascript
const net = await posenet.load();
```
By default, `posenet.load()` loads a faster and smaller model that is based on MobileNetV1 architecture and has a lower accuracy. If you want to load the larger and more accurate model, specify the architecture explicitly in `posenet.load()` using a `ModelConfig` dictionary:

#### MobileNet (smaller, faster, less accurate)
```javascript
const net = await posenet.load({
  architecture: 'MobileNetV1',
  outputStride: 16,
  inputResolution: { width: 640, height: 480 },
  multiplier: 0.75
});
```

#### ResNet (larger, slower, more accurate) \*\*new!\*\*
```javascript
const net = await posenet.load({
  architecture: 'ResNet50',
  outputStride: 32,
  inputResolution: { width: 257, height: 200 },
  quantBytes: 2
});
```
