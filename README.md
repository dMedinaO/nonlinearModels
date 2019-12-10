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



If you want send comments, opinion or you find a bug in library, please notify to via email: david.medina@cebib.cl
