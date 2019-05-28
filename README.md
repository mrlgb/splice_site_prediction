This repository is created for 2019 BME203 Intro to Bioinfo course individual project. It uses deep neural networks to predict splice site. 

## Dataset
This model uses the dataset specified in [HS3D](http://www.sci.unisannio.it/docenti/rampone/).

true and false splite site files shall be saved in ./data directory

## Dependencies
Pytorch for python 3.x should be installed.
To install all the other packages:
```
pip install ipdb, scikit-learn, tqdm, numpy
```

## How to Run
Simply run main.py script:
```
python main.py
```

## Reference
* [Naito T., (2018), Human Splice-Site Prediction with Deep Neural Networks, J Comput Biol., 25(8):954-961., doi: 10.1089/cmb.2018.0041](https://www.ncbi.nlm.nih.gov/pubmed/29668310)
* [ResNet pytorch implementation](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
