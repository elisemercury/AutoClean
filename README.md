# AutoClean

**Python Package for Automated Dataset Preprocessing & Cleaning**

:white_check_mark: The Duplicate Image Finder (difPy) Python package **automates** this task for you!

```python
pip install difPy
```

Read more on how the algorithm of difPy works in my Medium article [Automated Data Cleaning with Python](link).

## Description
It is commonly known among Data Scientists that data cleaning and preprocessing make up a major part of a data science project. And, in all honesty, on average it is not the most exciting part of the project.

:white_check_mark: AutoClean helps you automate major parts of these tasks and performs preprocessing in an automated manner.

AutoClean supports:

:point_right: various imputation methods for missing values

:point_right: handling of outliers

:point_right: encoding of categorical data (OneHot, Label)

:point_right: extraction of datatime values
 
:point_right: and more!

As an example, the following sample dataset will be passed through the AutoClean pipeline:

<p align="center">
  <img src="Misc/sample_data.png" width="300" title="Example Output: Duplicate Image Finder">
</p>

 The output of AutoClean looks as following, whereas the various adjustments have been highlighted:

 <p align="center">
  <img src="Misc/sample_data_output.png" width="500" title="Example Output: Duplicate Image Finder">
</p>