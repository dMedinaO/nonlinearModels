## README

RV-Clustering, a library of unsupervised learning algorithms, and a new methodology designed to find optimum partitions in highly non-linear datasets that allow deconvoluting variables and notoriously improving performance metrics in supervised learning classification or regression models.  The partitions obtained are statistically cross-validated,
ensuring correct representativity and no over-fitting. RV-Clustering has been successfully tested in several highly non-linear datasets with different origins. The approach herein proposed has generated classification and regression models with high-performance metrics, which further supports its ability to generate predictive models for highly non-linear datasets. Advantageously, the method does not require significant human input, which guarantees better usability in the biological, biomedical and protein engineering community with no specific knowledge in the machine learning area.

## Design and implementation

functionalities of the library were tested through the analysis of different datasets, mainly extracted from bibliographic reports of specific mutations in proteins and the effect they have on their properties and stability, and from open databases, such as BRENDA Jeske et al. (2018) and ProTherm Bava et al. (2004). A generalization mode is tested with different datasets extracted from the UCI Machine Learning repository Asuncion and Newman (2007).

## RV-Clustering modules


## 1. Checks Module

Module with all the characteristics associated with the revision of the data sets, linearity evaluation, among the main characteristics.

## 2. Graphic

Module that allow create a chart of results, apply differente visualizations for check data in distribution of performance and other elements.

## 3. Statistics analysis

Module with all the characteristics associated with statistics analysis in dataset.

## 4. Statistics corroboration

Module with all the characteristics associated with the revision of the data sets and statistics corroboration from split partitions and other characteristics.

## 5. Supervised learning classification

Module that allows to apply different supervised learning algorithms associated with classification tasks.

## 6. Supervised learning regression

Module that allows to apply different supervised learning algorithms associated with regression/prediction tasks.

## 7. Utils

Module with all characteristics  and functionalities associated to different requeriments for to work with dataset.

## Exec scripts and Recomendations of use

For to execute the script to evaluate if dataset is non-linear o linear, you can use the script: LauncherCheckNonLinear.py

```
usage: LauncherCheckNonLinear.py [-h] --dataset DATASET --type {1,2}
                                 [--threshold THRESHOLD] --feature FEATURE

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        dataSet input to check data
  --type {1,2}, -k {1,2}
                        Kind of dataSet: 1 for classification models --- 2 for
                        predictive models
  --threshold THRESHOLD, -t THRESHOLD
                        threshold to consider in linear evaluation of response
                        in dataset
  --feature FEATURE, -f FEATURE
                        Name of feature with response in dataset
```

In a second stage, if the evaluation of the dataset response that it is not linear, you can use the script for to explore different algorithms and parameters and
get performances. The scripts are: LauncherexploreRegressionModels.py | LauncherExploratoryClassifierModels.py, the use of each one, depends if response in dataset is
categorical/discrete or continue.

When response in dataset is continue type: you could to use LauncherexploreRegressionModels.py:

```
usage: LauncherexploreRegressionModels.py [-h] -d DATASET -p PATHRESULT -m
                                          PERFORMANCE -r RESPONSE -t THRESHOLD

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataSet DATASET
                        full path and name to acces dataSet input process
  -p PATHRESULT, --pathResult PATHRESULT
                        full path for save results
  -m PERFORMANCE, --performance PERFORMANCE
                        performance selected model
  -r RESPONSE, --response RESPONSE
                        name of column with response values in dataset
  -t THRESHOLD, --threshold THRESHOLD
                        threshold of minimus values expected form model
                        generated
```

In case of dataset has categorical or discrete response, you could use LauncherExploratoryClassifierModels.py:

```
usage: LauncherExploratoryClassifierModels.py [-h] -d DATASET -p PATHRESULT -m
                                              PERFORMANCE -r RESPONSE -t
                                              THRESHOLD -k KVALUEDATA

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataSet DATASET
                        full path and name to acces dataSet input process
  -p PATHRESULT, --pathResult PATHRESULT
                        full path for save results
  -m PERFORMANCE, --performance PERFORMANCE
                        performance selected model
  -r RESPONSE, --response RESPONSE
                        name of column with response values in dataset
  -t THRESHOLD, --threshold THRESHOLD
                        threshold of minimus values expected form model
                        generated
  -k KVALUEDATA, --kValueData KVALUEDATA
                        Value for cross validation, this value most be higher
                        or equal 2

```

If there are not any combination of parameter and algorithm that its performance is lower than threshold input. We recomender to use the methodology of splitter and training
differents groups partitions. For this, you can apply the scripts exposed then.

### 1. Encoding categorical features

For encoding categorical features, you can to use the scripts: LauncherEncodingClass.py, the output of scripts are: a dictionary in JSON format with values of categorical and its encoding. And, the dataset with categorical features changes by encoding process.

```
usage: LauncherEncodingClass.py [-h] -d DATASET -p PATH

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataSet DATASET
                        full path and name to acces dataSet input process
  -p PATH, --path PATH  path to save dataset transform

```

### 2. Split dataset using recursive binary strategy

For split dataset, you can to use the scripts: launcherClusteringScan.py, the output of scripts are: csv files with examples in divisions, dataset with categories associated to ID of groups, and representations of division in a image.

```
usage: launcherClusteringScan.py [-h] -d DATASET -p PATHRESULT -r RESPONSE -k
                                 {1,2} -t THRESHOLD -s INITIALSIZE

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataSet DATASET
                        full path and name to acces dataSet input process
  -p PATHRESULT, --pathResult PATHRESULT
                        full path for save results
  -r RESPONSE, --response RESPONSE
                        Name response in dataset
  -k {1,2}, --kind {1,2}
                        Kind of dataset: 1. Classification 2. Regression
  -t THRESHOLD, --threshold THRESHOLD
                        threshold for umbalanced class
  -s INITIALSIZE, --initialSize INITIALSIZE
                        initial Size of dataset
```

### 3. Training models for to classifier in partitions

After to create a partitions, you must to training a dataset for to classify examples in a group of partitions, for this, you can use the script LauncherTrainingModelPartitions.py

```
usage: LauncherTrainingModelPartitions.py [-h] -d DATASET -p PATHRESULT -r
                                          RESPONSE [-k KVALUEDATA]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataSet DATASET
                        full path and name to acces dataSet input process
  -p PATHRESULT, --pathResult PATHRESULT
                        full path for save results
  -r RESPONSE, --response RESPONSE
                        name of column with response values in dataset
  -k KVALUEDATA, --kValueData KVALUEDATA
                        Value for cross validation, this value most be higher
                        or equal 2
```

The script will create a model to classify examples in a class group, We recommender that you create a directory for save the results of this script.

NOTE: Please remember to use the dataset with categorical features encoding

### 4. To training different partitions using supervised learning algorithms

In the next stage, it is necessesary training models for all groups, created, for this, you can to use the scripts: LauncherTrainingModelPartitionsGroups.py

```
usage: LauncherTrainingModelPartitionsGroups.py [-h] -p PATHINPUT -r RESPONSE
                                                [-k KVALUEDATA] -t {1,2}

optional arguments:
  -h, --help            show this help message and exit
  -p PATHINPUT, --pathInput PATHINPUT
                        Path where is all data set created using clustering
                        partitions
  -r RESPONSE, --response RESPONSE
                        name of column with response values in dataset
  -k KVALUEDATA, --kValueData KVALUEDATA
                        Value for cross validation, this value most be higher
                        or equal 2
  -t {1,2}, --type {1,2}
                        Type of response in dataset: 1. class, 2. prediction
```

The performance are reported in standard output.

### 5. To get performance of meta-models

The finish stage is To get performance of meta-models, the performance are reported in standard output, for this, you can to use the scripts: LauncherUseValidationDataSet.py

```
usage: LauncherUseValidationDataSet.py [-h] -m MODELPARTITION -d DATASET -r
                                       RESPONSE -p PATHMODELS -t {1,2}

optional arguments:
  -h, --help            show this help message and exit
  -m MODELPARTITION, --modelPartition MODELPARTITION
                        Model created for to classify in a partition (*.joblib
                        file)
  -d DATASET, --dataset DATASET
                        Data set validation
  -r RESPONSE, --response RESPONSE
                        Name of feature response in dataset
  -p PATHMODELS, --pathModels PATHMODELS
                        Path models of partitions
  -t {1,2}, --type {1,2}
                        Type of model: 1. Class, 2. Regression

```

### 6. To use meta-models for predict new examples

Finally, you can use the models trainined for to classify new examples, for this reasson, you can use the script: LauncherUseModelsNewExamples.py, the predictions are reporte in standard output, and are encoding using the JSON file

```
usage: LauncherUseModelsNewExamples.py [-h] -m MODELPARTITION -d DATASET -p
                                       PATHMODELS -t {1,2}

optional arguments:
  -h, --help            show this help message and exit
  -m MODELPARTITION, --modelPartition MODELPARTITION
                        Model created for to classify in a partition (*.joblib
                        file)
  -d DATASET, --dataset DATASET
                        Data set validation
  -p PATHMODELS, --pathModels PATHMODELS
                        Path models of partitions
  -t {1,2}, --type {1,2}
                        Type of model: 1. Class, 2. Regression
  -j JSONENCODING --jsonencoding JSONENCODING
                        JSON File with encoding features applied in training model
```

If you want send comments, opinion or you find a bug in library, please notify to via email: david.medina@cebib.cl
