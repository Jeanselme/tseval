


tseval
======

tslearn is a Python module for evaluating the performance of machine learning models on time series tasks.

In its current version, this package is particularly useful to researchers in the medical domain due to the authors' bias. Furthermore, this package is particularly geared to time series data and not sequential data in general.



Why is time series evaluation different?
----------------------------------------

Time series tasks often highly differ from a regular classification or regression, say, classifying images into categories. Time series data is typically not i.i.d., i.e. each observation in the data is dependent on various factors, such as:
* the *time* at which the observation was taken
* the *time series* from which the observation is taken from, e.g. a particular patient when dealing with vital sign data or an article when dealing with NLP data

Most standard libraries in Python, such as scipy-learn or seaborn, do not provide curves or evaluation tools to deal with the non-i.i.d. case. Out of this lack of suitable open-source tools for evaluating time series models, this library was built.

Installation
------------

TODO dependencies -> libraries
TODO python interpreter

TODO pip installation -> https://packaging.python.org/tutorials/packaging-projects/


Documentation
-------------

TODO


Contribution
------------

Any contributions to this library, in particular from domains other than the medical domain, or suggestions on the existing code are very welcome!
Your contribution will be acknowledged.

Contributers
------------

The following people have actively developed the initial version of this library:

* Fabian Falck @ FabianFalck

The following people have contributed to the initial version of this library in terms of code, ideas and content:

* Vincent Jeanselme (Carnegie Mellon University)
* Peter Gyring (University of Oxford)
* Mauro Santos (University of Oxford)


Citation
--------

TODO

TODO
----

- see TODOs above
- in the documentation of each function, stress why it is different to the standard implementation of sklearn (e.g. for the ROC).
- rework:
    - ROC (split curve and plot)
    - make AMOC example with open source data
- metrics to add:
    - y pred over time
    - scalar metrics FPR, TPR
- scaler metrics: usage example

- in general: make for anything in the library a usage example
- no warranty clause in README


- for all functions: savefig or show option