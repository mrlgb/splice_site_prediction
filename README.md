This repository is created for 2019 BME203 Intro to Bioinfo course individual project. It uses ResNet to predict splice site. 

## Dataset
This model uses the dataset specified in [HS3D](http://www.sci.unisannio.it/docenti/rampone/).

true and false splice site files shall be saved in ./data directory

## Dependencies
Pytorch for python 3.x should be installed. Please follow instructions on [official page](https://pytorch.org/get-started/locally/).

To install all the other packages:
```
pip install ipdb, scikit-learn, tqdm, numpy
```

## How to Run
Simply run main.py script:
```
python main.py
```

## Test Session
To run test session, DSSP pretrained models should be stored in ./DSSP directory. Pretrained models can be downloaded in [github.com/DSSP-github/DSSP](https://github.com/DSSP-github/DSSP)

Please specify the saved model to test by using -output argument.
```
python test.py -output output_directory_name
```

## Reports
Several metrics were estimated on the balanced(1:1 ratio of labels) dataset.

| Model | Validation Acc | binary F1 | weighted F1 |
| resnet18-epoch15 | 82.870982 | 0.830501 | 0.828719 |
| DSSP | 49.229414 | 0.000000 | 0.324807 |

## Reference
* [Naito T., (2018), Human Splice-Site Prediction with Deep Neural Networks, J Comput Biol., 25(8):954-961., doi: 10.1089/cmb.2018.0041](https://www.ncbi.nlm.nih.gov/pubmed/29668310)
* [ResNet pytorch implementation](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
