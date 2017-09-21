# Machine Learning Engineer Nanodegree
## Specializations
## Project: Capstone Project

### Requirements

    Python >= 2.7.0
    numpy >= 1.11.1
    Keras >= 2.0.0
    
### Datasets 

The dataset ([download here](https://www.kaggle.com/c/integer-sequence-learning/data "Click to download dataset")) of this project contains the majority of the integer sequences from the On-Line Encyclopedia of Integer Sequences® (OEIS®). It is split into a training set, where you are given the full sequence, and a test set, where we have removed the last number from the sequence. The task is to predict this removed integer.

Note that some sequences may have identical beginnings (or even be identical altogether). They have not been removed these from the dataset.

#### File descriptions
- train.csv - the training set, contains full sequences
- test.csv - the test set, missing the last number in each sequence

### Running Flow
- Step 1: Download dataset and put in `data` folder
- Step 2: Run `prepro.py` to preprocess data
- Step 3: Run `train.py` to train data
- Step 4: Run `result.py` to get the final prediction result `result.csv`
 
