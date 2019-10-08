## README

RV-Clustering, a library of unsupervised learning algorithms, and a new methodology designed to find optimum partitions in highly non-linear datasets that allow deconvoluting variables and notoriously improving performance metrics in supervised learning classification or regression models.  The partitions obtained are statistically cross-validated,
ensuring correct representativity and no over-fitting. RV-Clustering has been successfully tested in several highly non-linear datasets with different origins. The approach herein proposed has generated classification and regression models with high-performance metrics, which further supports its ability to generate predictive models for highly non-linear datasets. Advantageously, the method does not require significant human input, which guarantees better usability in the biological, biomedical and protein engineering community with no specific knowledge in the machine learning area.

## Design and implementation

Both the source code and the executable elements of RV-Clustering were implemented under the Python 2.7 programming language Oliphant (2007), mainly using the Scikit-learn Pedregosa et al. (2011), Python Data Analysis (Pandas) McKinney (2011) and NumPy Van Der Walt et al. (2011) libraries.  The RV-Clustering library was designed under the Object Oriented Programming paradigm Wegner (1990), aiming to provide the modularity required to perform actions separately in the proposed workflow. The different
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


## NOTES

If you want send comments, opinion or you find a bug in library, please notify to via email: david.medina@cebib.cl
