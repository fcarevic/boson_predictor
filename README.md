# Machine Learning Project - The Higgs Boson <!-- omit in toc -->


This team project is a part of the [Machine Learning](<https://www.epfl.ch/labs/mlo/machine-learning-cs-433> "EPFL's Machine Learning course") curriculum at the EPFL.

The purpose of this file is to explain the project/code structure and to help you running the scripts. 

For more information about the implementation, feel free to check out the commented code, as well as the [final report](report/Machine_Learning_Project_1.pdf), which contains the entire thought process, with all findings and conclusions.

---

## Table of Contents <!-- omit in toc -->
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Code structure](#code-structure)
- [Authors](#authors)

---

## Introduction

The Higgs boson is an elementary particle in the Standard Model of physics which explains why other particles
have mass. The aim of this project is to apply machine learning techniques to actual CERN particle accelerator data in order to recreate the process of "discovering” the Higgs particle. To be more precise, the ultimate goal is to estimate the likelihood that a given event’s signature was the result of a Higgs boson (signal) or some other process/particle (background).

---

## Getting Started

As a prerequisite, **python3** and **pip3** should already be installed.

1. Install **numpy**

  ```
  pip3 install numpy
  ```

2. Navigate to the `scripts/` folder of this project

  ```
  cd scripts
  ```

3. Create `data/` folder inside the root directory of the repository. Place the `train.csv` and `test.csv` files (they can be obtained from [here](https://www.kaggle.com/c/epfml18-higgs/data)) inside `data/`.

4. Run the following command:

  ```
  python3 run.py
  ```

---

## Code structure

### `run.py` <!-- omit in toc -->

This file was set up so it generates the best predictions for the project.

---

### `run_full_training.py` <!-- omit in toc -->

Training script for the best predictions for the project.
Note that trained models will not be the same, but will have same overall accuracy and fscore on the test data.

---

### `implementations.py` <!-- omit in toc -->

Implementations of following regression algorithms:

* **gradient descent**
* **stochastic gradient descent**
* **least squares** using normal equations
* **ridge regression** using normal equations
* **logistic regression** using GD or SGD
* **regularized logistic regressiong** using GD or SGD

*These implementations were a project requirement.*

---

### `preprocessing.py` <!-- omit in toc -->

All functions needed for the data preprocessing can be found in this file. These functions ensure:

* data cleaning
  * NaN values replacement (by mean/median)
  * outliers removal/clipping (Tukey's method)
* data splitting based on <em>pri_jet_num</em> feature
* PCA components extraction
* standardization
* data splitting (test/train)

---

### `featurization.py` <!-- omit in toc -->

This file ensures the following feature manipulation:

* polynomial feature expansion
* one-hot encoding of categorical features

---

### `genetic_algorithm.py` <!-- omit in toc -->

This file provides a full implementation (e.g. mutation, selection, cross-over) of the genetic algorithm, which is used for tuning and finding optimal hyperparameters.

---

### `hyperoptim.py` <!-- omit in toc -->

This file provides an implementation of a random search algorithm that is used for tuning and finding optimal hyperparameters.

---

### `proj1_helpers.py` <!-- omit in toc -->

Function helpers are included here, such as:

* csv file reader
* csv submission creator

---

### `visualization.py` <!-- omit in toc -->

This file is used for visualization (descriptive statistics) purposes (e.g. plotting PCA components on a scatter plot).

---

### `project1.ipynb` <!-- omit in toc -->

The notebook combines all the previously mentioned functions with the aim to analyze and process data, as well as to apply models and tune hyperparameters.

---

## Authors

* Carevic Filip
* Maid Edvin
* Mitic Natalija