# Detect human parts (PARTNet)
[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)]()
[![Status badge](https://img.shields.io/badge/status-in%20progress-green.svg)]()
[![Type badge](https://img.shields.io/badge/type-compact-blueviolet.svg)]()

<p align="center">
  <img src="banner.png" width="100%"/>
</p>

## What is PARTNet
PARTNet is a compact model trained to be faster and smaller.
The initial idea of this project is train a model to detect human head parts and expand to the entire human body.
Our model is trained over [ONNX](http://onnx.ai), this allows the model to be executed in real time on mobile devices and embedded devices, it also allows the model to be converted to other neural network architectures.
The model is not as accurate as a full model, but it's being constantly updated for best results.

## What is ONNX?
The Open Neural Network eXchange ([ONNX](http://onnx.ai)) is an open format to represent deep learning models. With ONNX, developers can move models between state-of-the-art tools and choose the combination that is best for them. ONNX is developed and supported by a community of partners.

## Model Visualization
You can see visualizations of each model's network architecture by using [Netron](https://lutzroeder.github.io/Netron) or [VisualDL](http://visualdl.paddlepaddle.org/).

### Run samples
We offer the model in 3 different formats: ONNX, CoreML and TensorFlow. Download according to the last registered timestamp, the format is: **dd_mm_yyyy__hh_mm_ss**. To execute them simply unzip the specific file and run the command:

```bash
python python/onnxruntime_predict.py <image_file>
```
or

```bash
python python/predict.py <image_file>
```

[![Visualizer badge](https://img.shields.io/badge/visualizer-netron-blue.svg)](https://lutzroeder.github.io/Netron)
[![Visualizer badge](https://img.shields.io/badge/visualizer-visualdl-blue.svg)](http://visualdl.paddlepaddle.org/)

## Statistics
[![Legend badge](https://img.shields.io/badge/-precision-blue.svg)]()
[![Legend badge](https://img.shields.io/badge/-recall-orange.svg)]()
[![Legend badge](https://img.shields.io/badge/-mAP-green.svg)]()

<p align="center">
<img src="/statistics/precision.png" width="30%"/>
<img src="/statistics/recall.png" width="30%"/>
<img src="/statistics/mAP.png" width="30%"/>
</p>

These are the statistics of the last iteration performed. They are based on a subsample of the original image database that contains 60,000 images.

### Performance Per Tag

| Tag | Precision | Recall | A.P. | Image Count |
| --- | --------- | ------ | ---- | ----------- |
| Human beard | 78.6% | 10.8% | 34.9% | 5044 |
| Man | 38.3% | 1.8% | 13.3% | 5087 |
| Human hair | 33.3% | 0.0% | 6.5% | 5105 |
| Woman | 30.4% | 0.2% | 10.7% | 5059 |
| Human head | 20.0% | 0.0% | 5.9% | 5123 |
| Human ear | 11.8% | 0.1% | 4.8% | 5071 |
| Human eye | 11.1% | 0.0% | 5.5% | 5115 |
| Human nose | 0.0% | 0.0% | 3.1% | 5121 |
| Human face | 0.0% | 0.0% | 6.9% | 5090 |
| Human eyebrow | 0.0% | 0.0% | 5.0% | 5081 |
| Human forehead | 0.0% | 0.0% | 5.3% | 5094 |
| Human mouth | 0.0% | 0.0% | 3.8% | 5116 |

## Dataset
We use a subset of images extracted from Google Open Images Dataset V5 ([OIDV5](https://storage.googleapis.com/openimages/web/factsfigures.html)) containing 60,000 images that were randomly extracted using the [OIDv4_ToolKit tool](https://github.com/EscVM/OIDv4_ToolKit). 
The database is really great, if you have interest contact us by [email](mailto:claudemir.casa@ufpr.br).

## Contributions
Do you want to contribute a model? To get started, pick any model presented above with the [contribute]() link under the Description column. The links point to a page containing guidelines for making a contribution.

## License
[![License badge](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

MIT License. Copyright (c) 2019 IMAGO Research Group.

## Authors
[Claudemir Casa](claudemir.casa)
