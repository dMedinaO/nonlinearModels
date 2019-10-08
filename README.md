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

## Exec scripts

For to execute the script to evaluate if dataset is non-linear o linear, the command line is

```
python LauncherCheckNonLinear.py dataSet tipo featureResponse threshold
```

Where dataset is full path to acces dataset input, tipo is kind of data set 1 classification task, any value for regression, featureResponse is name columne in csv with values for response and threshold is the value of threshold for to evaluate performance data.

For to explore supervised learning algorithms, you can run the scripts LauncherExploratoryClassifierModels.py for classification task and LauncherexploreRegressionModels for predictive task, in both case the parameters for execution are similar:

```
python script.py -d dataset -p pathResult -m performance -r response -t threshold
```

The parameters are:

-d or --dataSet is a full path and name to acces dataSet input process
-p or --pathResult is a full path for save results
-m or --performance is a performance selected model
-r or --response is a name of column with response values in dataset
-t or --threshold is a threshold of minimus values expected form model generated

Finally for to create a multi split data with recursive binary method you caun use:

```
python launcherClusteringScan.py [-h] -d DATASET -o OPTION -p PATHRESULT -r
                                 RESPONSE -k KIND -t THRESHOLD -s INITIALSIZE

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataSet DATASET
                        full path and name to acces dataSet input process
  -o OPTION, --option OPTION
                        Option to Normalize data set: 1. Normal Scale 2. Min
                        Max Scaler 3. Log scale 4. Log normal scale
  -p PATHRESULT, --pathResult PATHRESULT
                        full path for save results
  -r RESPONSE, --response RESPONSE
                        Name response in dataset
  -k KIND, --kind KIND  Kind of dataset: 1. Classification 2. Regression
  -t THRESHOLD, --threshold THRESHOLD
                        threshold for umbalanced class
  -s INITIALSIZE, --initialSize INITIALSIZE
                        initial Size of dataset
```


## NOTES

If you want send comments, opinion or you find a bug in library, please notify to via email: david.medina@cebib.cl
