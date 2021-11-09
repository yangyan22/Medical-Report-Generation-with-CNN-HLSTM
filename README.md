Medical Image Report Generation with CNN-HLSTM: A basic model for medical image report generation. The CNN is used to encode visual features and the hierachical LSTM (i.e., two LSTM networks) is adopted to decode the visual features and generate the report. 

To perform the evaluation of the metric Meteor, you need to download the package "paraphrase-en.gz" and put it to the path "./pycocoevalcap/meteor/data/paraphrase-en.gz". We did not upload it since the file was too big. 


If you find the code useful, please cite our paper "Joint Embedding of Deep Visual and Semantic Features for Medical Image Report Generation", IEEE Transactions on Multimedia, 2021. 

# Dependencies
  - Python=3.7.3
  - pytorch=1.8.1
  - pickle
  - tqdm
  - time
  - argparse
  - matplotlib
  - sklearn
  - json
  - numpy 
  - torchvision 
  - itertools
  - collections
  - math
  - os
  - matplotlib
  - PIL 
  - itertools
  - copy
  - re
  - abc
  - pandas
  - torch

# reference codes: 
https://github.com/cuhksz-nlp/R2Gen

https://github.com/tylin/coco-caption
