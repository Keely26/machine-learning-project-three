CSCI 447 — Machine Learning: Soft Computing
Project #3
Assigned: October 16, 2017
Design Document Due: October 30, 2017
Project Due: November 13, 2017

This assignment requires you to apply three different evolutionary approaches to train a feedforward
neural network and then compare the results of evolutionary training to backpropagation. For this particular
assignment, you will focus only on training the weights of a pre-defined neural network, thus the only
difference between the various implementations for a given problem will be the training algorithm.
Starting with an untrained neural network, you will train the weights of the network using each of the
following algorithms (separately):

• Backpropagation
• (µ + λ)-evolution strategy
• Differential evolution
• Genetic algorithm with real-valued chromosomes

For each of these algorithms, you will conduct appropriate studies to determine the best parameter settings
(e.g., learning rate, momentum, µ, λ) prior to running the head-to-head comparisons. Be sure to apply good
experimental design in your process (e.g., tuning sets, cross-validation, statistical hypothesis testing).
Your assignment consists of the following steps:
  1. Prepare a design document addressing the design of three different neural network training methods
    applied to a fixed structure neural network to be used either for classification or function approximation
    (see below). Be sure to include an explanation of your experimental design as well.
  2. Implement feedforward neural network training using a tunable evolution strategy.
  3. Implement feedforward neural network training using a tunable differential evolution procedure.
  4. Implement feedforward neural network training using a tunable genetic algorithm.
  5. Develop a hypothesis focusing on convergence rate and final performance of each of the chosen algorithms.
  6. Compare the results of all three of your algorithms and backpropagation (which should have been
    implemented previously), after tuning, on five (5) different classification or function approximation
    problems of your choosing from the UCI Machine Learning Repository
    (http://archive.ics.uci.edu/ml/datasets.html).
  7. Write a paper that incorporates the following elements, summarizing the results of your experiments:
    (a) Title and author name
    (b) A brief, one paragraph abstract summarizing the results of the experiments
    (c) Problem statement, including hypothesis
    (d) Description of algorithms implemented
    (e) Description of your experimental approach
    (f) Presentation of the results of your experiments
    (g) A discussion of the behavior of your algorithms, combined with any conclusions you can draw
    (h) Summary
    (i) References (you should have at least one reference related to each of the algorithms implemented
        and any other references you consider to be relevant)
  8. Submit your design document, fully documented code, sample runs of each algorithm, and your paper.
