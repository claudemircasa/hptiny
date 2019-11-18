# Detect and segment human body parts
[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)]()
[![Status badge](https://img.shields.io/badge/status-in%20progress-green.svg)]()
[![Type badge](https://img.shields.io/badge/type-compact-blueviolet.svg)]()
[![Version badge](https://img.shields.io/badge/v-1.3.0-blue.svg)]()

<p align="center">
  <img src="7ba73e53-c0c8-4969-9cfe-2f8a4c8ad009.jpg" width="100%"/>
</p>

<img src="/icons/person.png" width="7.8%"/> <img src="/icons/man.png" width="7.8%"/> <img src="/icons/woman.png" width="7.8%"/> <img src="/icons/boy.png" width="7.8%"/> <img src="/icons/girl.png" width="7.8%"/> <img src="/icons/head.png" width="7.8%"/> <img src="/icons/face.png" width="7.8%"/> <img src="/icons/hair.png" width="7.8%"/> <img src="/icons/beard.png" width="7.8%"/> <img src="/icons/nose.png" width="7.8%"/> <img src="/icons/eyebrow.png" width="7.8%"/> <img src="/icons/eye.png" width="7.8%"/> <img src="/icons/ear.png" width="7.8%"/> <img src="/icons/leg.png" width="7.8%"/> <img src="/icons/foot.png" width="7.8%"/> <img src="/icons/arm.png" width="7.8%"/> <img src="/icons/hand.png" width="7.8%"/> <img src="/icons/mouth.png" width="7.8%"/>

## Lastest Releases

[v1.4.3-alpha](https://github.com/claudemircasa/hptiny/releases/tag/v1.4.3-alpha)
[v1.4.2-alpha](https://github.com/claudemircasa/hptiny/releases/tag/v1.4.2-alpha)

## What is hptiny
hptiny is a compact model trained to be faster and smaller.
The initial idea of this project is train a model to detect human body visible parts and expand to specific regions.
Our model is trained over [ONNX](http://onnx.ai), this allows the model to be executed in real time on mobile devices and embedded devices, it also allows the model to be converted to other neural network architectures.
The model is not as accurate as a full model, but it's being constantly updated for best results.

## What is ONNX?
The Open Neural Network eXchange ([ONNX](http://onnx.ai)) is an open format to represent deep learning models. With ONNX, developers can move models between state-of-the-art tools and choose the combination that is best for them. ONNX is developed and supported by a community of partners.

## Model Visualization
You can see visualizations of each model's network architecture by using [Netron](https://lutzroeder.github.io/Netron) or [VisualDL](http://visualdl.paddlepaddle.org/).

### Run samples
We provide compressed binary versions of the model. To execute them simply unzip the specific file and run the command to see a list of valid options:

```bash
python hptiny.py
```

[![Visualizer badge](https://img.shields.io/badge/visualizer-netron-blue.svg)](https://lutzroeder.github.io/Netron)
[![Visualizer badge](https://img.shields.io/badge/visualizer-visualdl-blue.svg)](http://visualdl.paddlepaddle.org/)

## Statistics

These are the statistics of the last iteration performed. They are based on a subsample of the original image database that contains 127,000 images.

### Performance Per Tag mAP@IoU=50

| images | detections count | unique truth count |
| - | - | - |
| 13032 | 555774 |54664 | 

| class id | name | ap | TP | FP |
| - | - | - | - | - |
| 0 | Person | 0.00% | 0 | 0 |
| 1 | Man | 19.17% | 58 | 32 |
| 2 | Woman | 22.49% | 64 | 32 |
| 3 | Boy | 39.52% | 238 | 54 |
| 4 | Girl | 24.19% | 90 | 46 |
| 5 | Human head | 10.30% | 46 | 160 |
| 6 | Human face | 14.89% | 60 | 148 |
| 7 | Human eye | 15.49% | 18 | 28 |
| 8 | Human eyebrow | 10.91% | 4 | 20 |
| 9 | Human nose | 7.98% | 14 | 34 |
| 10 | Human mouth | 12.60% | 24 | 30 |
| 11 | Human ear | 16.51% | 136 | 182 |
| 12 | Human hair | 9.17% | 40 | 170 |
| 13 | Human beard | 46.13% | 460 | 236 |
| 14 | Human leg | 19.81% | 114 | 56 |
| 15 | Human arm | 10.99% | 32 | 16 |
| 16 | Human foot | 30.47% | 60 | 0 |
| 17 | Human hand | 11.30% | 64 | 30 |

for conf_thresh = 0.25, precision = 0.54, recall = 0.03, F1-score = 0.05 
for conf_thresh = 0.25, TP = 1522, FP = 1274, FN = 53142, average IoU = 40.30 %

IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
mean average precision (mAP@0.50) = 0.178847, or 17.88 % 
Total Detection Time: 106.000000 Seconds

## Dataset
We use a subset of images extracted from Google Open Images Dataset V5 ([OIDV5](https://storage.googleapis.com/openimages/web/factsfigures.html)) containing 127,000 images that were randomly extracted using the [OIDv4_ToolKit tool](https://github.com/EscVM/OIDv4_ToolKit). 
The database is really great, if you have interest contact us by [email](mailto:claudemir.casa@ufpr.br).

## Contributions
Do you want to contribute? To get started, choose the latest version of the template above, retrain, if you get better results with the same image database it will be published here.

## License
[![License badge](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

MIT License. Copyright (c) 2019 IMAGO Research Group.

## Authors
[Claudemir Casa](claudemir.casa)

## Collaborators
Special thanks to my lab colleagues.

[Jhonatan Souza](https://github.com/xkiddie)
[Tiago Mota de Oliveira](https://github.com/tiufsc)
