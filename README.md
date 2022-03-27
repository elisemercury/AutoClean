# AutoClean - Automated Data Preprocessing & Cleaning

![PyPI](https://img.shields.io/pypi/v/py-AutoClean)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/py-AutoClean)
![PyPI - License](https://img.shields.io/pypi/l/py-AutoClean)
![Medium](https://img.shields.io/badge/-Medium-000000?logo=Medium&logoColor=white)

**AutoClean automates data preprocessing & cleaning in your next Data Science project.**

```python
pip install py-AutoClean
```
:thought_balloon: Read more on how the algorithm of AutoClean works in my Medium article [Automated Data Cleaning with Python](link).

---

## Description
It is commonly known among Data Scientists that data cleaning and preprocessing make up a major part of a data science project. And, you will probably agree with me that it is not the most exciting part of the project. *Wouldn't it be great if this part could be automated?*

:white_check_mark: AutoClean helps you exactly with that: it performs **preprocessing** and **cleaning** of data in an **automated manner**, so that you can **save time** when working on your next project.

AutoClean supports:

:point_right: Various imputation methods for **missing values**  
:point_right: Handling of **outliers**  
:point_right: **Encoding** of categorical data (OneHot, Label)  
:point_right: **Extraction** of datatime values  
:point_right: and more!

## Basic Usage

AutoClean takes a Pandas dataframe as input and has a built-in logic of how to *automatically* clean and process your data. You can let your dataset run through the default AutoClean pipeline by using:

````python
from AutoClean import AutoClean
pipeline = AutoClean(dataset)
````

The resulting output dataframe can be accessed by using:

````python
pipeline.output

> Output:
    col_1  col_2  ...  col_n
1   data   data   ...  data
2   data   data   ...  data
... ...    ...    ...  ...
````

### Example
As a visual example, the following sample dataset will be passed through the AutoClean pipeline:

<p align="center">
  <img src="Misc/sample_data.png" width="300" title="Example Output: Duplicate Image Finder">
</p>

 The output of AutoClean looks as following, whereas the various adjustments have been highlighted:

 <p align="center">
  <img src="Misc/sample_data_output.png" width="700" title="Example Output: Duplicate Image Finder">
</p>

## Adjustable Parameters

In some cases, the default settings of AutoClean might not optimally fit your data. Therefore it also supports manual settings so that you can adjust it to whatever processing you might need. 

It has the following adjustable parameters, for which the options and descriptions can be found below:

````python
AutoClean(dataset, missing_num='auto', missing_categ='auto', encode_categ=['auto'],     
          extract_datetime='s', outliers='winz', outlier_param=1.5, logfile=True, verbose=False)
````

| Parameter | Type | Default Value | Other Values |
| ------ | :---: | :---: | ------ | 
| missing_num | `str` | `'auto'` | `linreg`, `knn`, `mean`, `median`, `most_frequent`, `delete` |
| missing_categ | `str` | `'auto'` | `logreg`, `knn`, `most_frequent`, `delete` |
| missing_categ | `list` | `['auto']` | `['onehot']`, `['label']`; to encode only specific columns add a list of column names or indexes: `['auto', ['col1', 2]]` |
| extract_datetime | `str` | `'s'` | `D`, `M`, `Y`, `h`, `m` |
| outliers | `str` | `'winz'` | `delete`|
| outlier_param | `int`, `float` | `1.5` | any int or float |
| logfile | `bool` | `True` | `False` |
| verbose | `bool` | `False` | `True` |

### missing_num

Defines how **numerical** missing values in the data are handled. Missing values can be predicted, imputed or deleted. When set to `auto`, AutoClean first attempts to predict the missing values with **Linear Regression**, and the values that could not be predicted are **imputed with K-NN**.

You can specify the handling method by setting `missing_num` to: `'linreg'`, `'knn'`, `'mean'`, `'median'`, `'most_frequent'` or `'delete'`.

### missing_categ

Defines how **categorical** missing values in the data are handled. Missing values can be predicted, imputed or deleted. When set to `auto`, AutoClean first attempts to predict the missing values with **Logistic Regression**, and the values that could not be predicted are **imputed with K-NN**.

You can specify the handling method by setting `missing_categ` to: `'logreg'`, `'knn'`, `'most_frequent'` or `'delete'`.

### extract_datetime

AutoClean can search the data for datetime features, and extract the values to separate columns. You csan set the granularity of the 

### outliers

### outlier_param

### logfile

### verbose