# Deep Learning for Economics and Finance (February 26 -- 28 & March 25 -- 27, 2025)

This is an mini-course on "Deep Learning for Economics and Finance". 
Its hels in two sessions:

* Session 1: Wednesday, February 26th -- Friday, February 28th, 2025
* Session 2: Tuesday, March 25th -- Thursday, March 27th, 2025


## Purpose of the lectures

* This mini-course is designed for Ph.D. students in economics and related disciplines. It
introduces recent advancements in applied mathematics, machine learning, computational
science, and the computational economics literature. The course focuses on solving and
estimating dynamic stochastic economic models, performing parametric uncertainty quantification, and modeling sequence data using recurrent neural networks, LSTMs, and
transformers.
* The lectures will concentrate on machine learning methodologies, including Deep Neural
Networks, Gaussian Processes, advanced architectures like RNNs, LSTMs, Transformers, and Generative AI models. These methods will be explored through applications in
macroeconomics, finance, and climate-change economics.
* The format of the lectures will be interactive and workshop-like, combining theoretical
discussions with hands-on coding exercises. The coding will be conducted in Python and
implemented on a cloud computing infrastructure


## Prerequisites

* Basic econometrics.
* Basic programming in Python (see [this link to QuantEcon](https://python-programming.quantecon.org/intro.html) for a thorough introduction).
* A brief Python refresher is provided [under this link](python_refresher).
* A brief Python on Jupyter Notebooks is provided [under this link](python_refresher/jupyter_intro.ipynb). 
* Basic calculus and probability (The book [Mathematics for Machine learning](https://mml-book.github.io/) provides a good overview of skills participants are required to be fluent in). 


## Class enrollment on the [Nuvolos Cloud](https://nuvolos.cloud/)

* All lecture materials (slides, codes, and further readings) will be distributed via the [Nuvolos Cloud](https://nuvolos.cloud/).
* To enroll in this class, please click on this [enrollment key](https://app.nuvolos.cloud/enroll/class/qMepeoToZXg), and follow the steps.


### Novolos Support

- Nuvolos Support: <support@nuvolos.cloud>


## Topics

### Session 1 

### [Day 1](lectures/day1), Wednesday, February 26th, 2025 (Pavillon Mail PM03)

 **Time** | **Main Topics** 
------|------
09:00 - 10:30 | [Introduction to Machine Learning and Deep Learning (part I)](lectures/day1/slides/01_Intro_to_DeepLearning.pdf) 
10:30 - 10:45 | Coffee Break
10:45 - 12:15 | [Introduction to Machine Learning and Deep Learning (part II)](lectures/day1/slides/01_Intro_to_DeepLearning.pdf) 

### [Day 2](lectures/day2), Thursday, February 27th, 2025 (Pavillon Mail PM03)

 **Time** | **Main Topics** 
------|------
09:00 - 10:30 | [A hands-on session on Deep Learning, Tensorflow, and Tensorboard](lectures/day1/code) 
10:30 - 10:45 | Coffee Break
10:45 - 12:15 | [Introduction to Deep Equilibrium Nets (DEQN)](lectures/day1/slides/01_DeepEquilibriumNets.pdf) 
12:15 - 13:30 | Lunch Break 
13:30 - 14:15 | Hands-on: Solving a dynamic model with [DEQNs](lectures/day2/code/02_Brook_Mirman_1972_DEQN.ipynb) 
14:15 - 15:00 | Hands-on: Solving a dynamic stochastic model with [DEQNs](lectures/day2/code/02_Brock_Mirman_Uncertainty_DEQN.ipynb) 
15:00 - 15:15 | Coffee Break
15:15 - 16:00 | Exercise: Solving a dynamic stochastic model by [example](lectures/day2/code/03_DEQN_Exercises_Blancs.ipynb) 
16:00 - 16:45 | [Introduction to a tuned DEQN library](lectures/day2/code/DEQN_production_code): [solving a stochastic dynamic OLG model with an analytical solution](lectures/day2/slides/02_OLG_with_analytical_solution_model.pdf) 

### [Day 3](lectures/day3), Friday, February 28th, 2025 (Pavillon Mail PM03)

 **Time** | **Main Topics** 
------|------
09:00 - 10:30 | [Surrogate models part I:](lectures/day3/slides/01_Surrogate_models.pdf) (for structural estimation and uncertainty quantification via [deep surrogate models](lectures/day2/readings/Deep_Surrogates.pdf)), with an example [DSGE model solved with DEQN and pseudo-states](lectures/day3/code/DEQN_production_code/stochastic_growth_pseudostates) 
10:30 - 10:45 | Coffee Break
10_45 - 12:15 | [Surrogate models part II:](lectures/day3/slides/02_Surrogate_models.pdf) (for structural estimation and uncertainty quantification via [Gaussian process regression](lectures/day2/readings/Machine_learning_dynamic_econ.pdf) 


### Session 2

### [Day 4](lectures/day4), Tuesday, March 25th, 2025 (Uni Dufour U364)

 **Time** | **Main Topics** 
------|------
09:00 - 10:30 | Bayesian Active Learning and GPs for Surrogate Models
10:30 - 10:45 | Coffee Break
10:45 - 12:15 | [Creating GP-based surrogates from DSGE models](lectures/day4/code/DEQN_production_code) 

### [Day 5](lectures/day5), Wednesday, March 26th, 2025 (Uni Dufour, Salle 408)

 **Time** | **Main Topics** 
------|------
09:00 - 09:45 | [Introduction to the macroeconomics of climate change, and integrated assessment models](lectures/day5/slides/01_Intro_IAM.pdf) 
09:45 - 10:30 | [Solving dynamic stochastic, nonlinear, nonstationary models, with an application to Integrated Assessment Models](lectures/day5/slides/01_Intro_IAM.pdf)
10:30 - 10:45 | Coffee Break
12:15 - 13:00 | [Solving the (non-stationary) DICE model](lectures/day3/slides/02_b_CDICE_implementation.pdf) with [Deep Equilibrium Nets](lectures/day5/code/DEQN_for_IAMs) 
12:30 - 13:30 | Lunch Break 
13:30 - 15:00 | Putting things together: [Deep Uncertainty Quantification for stochastic integrated assessment models](lectures/day5/slides/03_Deep_UQ_IAM.pdf)
15:00 - 15:15 | Coffee Break
15:15 - 16:45 | Modeling Sequence Data with RNNs, LSTMs, and Transformers

### [Day 6](lectures/day6), Thursday, March 27th, 2025 (Uni Dufour U365)
 **Time** | **Main Topics** 
------|------
09:00 - 10:30 | Introduction to Generative AI in Economics and Finance
10:30 - 10:45 | Coffee Break
10:45 - 12:15 | Course Wrap-up and Future Directions


### Teaching philosophy
Lectures will be interactive, in a workshop-like style,
using [Python](http://www.python.org), [scikit learn](https://scikit-learn.org/), [Tensorflow](https://www.tensorflow.org/), and
[Tensorflow probability](https://www.tensorflow.org/probability) on [Nuvolos](http://nuvolos.cloud),
a browser-based cloud infrastructure in which files, datasets, code, and applications work together,
in order to directly implement and experiment with the introduced methods and algorithms.


### Lecturer
- [Simon Scheidegger](https://sites.google.com/site/simonscheidegger/) (HEC, University of Lausanne)


## Citation

Please cite [Deep Equilibrium Nets](https://onlinelibrary.wiley.com/doi/epdf/10.1111/iere.12575), [The Climate in Climate Economics](https://academic.oup.com/restud/advance-article-abstract/doi/10.1093/restud/rdae011/7593489?redirectedFrom=fulltext&login=false), and [Deep surrogates for finance: With an application to option pricing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3782722) in your publications if this repository helps your research:

```
@article{https://doi.org/10.1111/iere.12575,
author = {Azinovic, Marlon and Gaegauf, Luca and Scheidegger, Simon},
title = {DEEP EQUILIBRIUM NETS},
journal = {International Economic Review},
volume = {63},
number = {4},
pages = {1471-1525},
doi = {https://doi.org/10.1111/iere.12575},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/iere.12575},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/iere.12575},
year = {2022}
}
```

```
@article{10.1093/restud/rdae011,
    author = {Folini, Doris and Friedl, Aleksandra and Kübler, Felix and Scheidegger, Simon},
    title = "{The Climate in Climate Economics*}",
    journal = {The Review of Economic Studies},
    pages = {rdae011},
    year = {2024},
    month = {01},
    issn = {0034-6527},
    doi = {10.1093/restud/rdae011},
    url = {https://doi.org/10.1093/restud/rdae011},
    eprint = {https://academic.oup.com/restud/advance-article-pdf/doi/10.1093/restud/rdae011/56663801/rdae011.pdf},
}
```

```
@article{chen2023deep,
  title={Deep surrogates for finance: With an application to option pricing},
  author={Chen, Hui and Didisheim, Antoine and Scheidegger, Simon},
  journal={Available at SSRN 3782722},
  year={2023}
}
```


# Auxiliary materials 

| Session #        |  Title     | Screencast  |
|:-------------: |:-------------:| :-----:|
|   1 	|First steps on Nuvolos | <iframe src="https://player.vimeo.com/video/513310246" width="640" height="400" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe>|
|   2 	| Terminal intro | <iframe src="https://player.vimeo.com/video/516691661" width="640" height="400" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe>|
