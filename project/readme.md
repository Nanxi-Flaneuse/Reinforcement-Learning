# Comparison of Reinforcement Learning Techniques for Movie Recommendations

## Authors
Nicole Falicov, Nanxi Liu, Angela Shen 

## Project Overview
This project explores the development and comparison of two reinforcement learning (RL) methods for generating successful personalized movie recommendations. Unlike traditional classification approaches, RL techniques are utilized here to maximize long-term user engagement and satisfaction by handling dynamic environments effectively.

## Methods
The project implements and compares the following two RL architectures:

* **SlateQ**: A decomposition of value-based temporal-difference and Q-learning designed to make RL tractable for "slates" (lists of recommendations) rather than single items. It uses a neural network to approximate Q-values for user-item pairs based on concatenated user and movie embeddings.
* **Double Q-Learning (Double DQN)**: An extension of the Deep Q-Network (DQN) that uses a target network and experience replay. It employs a second set of weights to reduce overoptimistic value estimates, leading to more stable learning outcomes.

## Dataset
Experiments were conducted using the **MovieLens Dataset** (Harper & Konstan, 2015) in two scales:
* **MovieLens-100k**: 100,000 ratings from 943 users on 1,682 movies.
* **MovieLens-1M**: 1,000,000 ratings from 6,000 users on 4,000 movies.
Data was split using an 80/20 train-test ratio based on chronological rating timestamps to assess future viewing predictions.

## Evaluation Metrics
The models were evaluated against random baselines using the following metrics:
* **Top-k Hit Rate**: Measures the percentage of users for whom at least one recommended item was watched in the future (k=4).
* **Normalized Discounted Cumulative Gain (nDCG@4)**: Evaluates the accuracy of the recommendation rankings and predicted ratings.

## Key Results
* **SlateQ Performance**: SlateQ performed approximately three times better than the random baseline, achieving a top-4 hit rate of **18%** and an nDCG of **12%** on the 100k dataset. Performance scaled down slightly on the 1M dataset to a 10% hit rate.
* **Double DQN Performance**: Results were lower than anticipated, with a peak top-4 accuracy of **3.4%** on the 100k dataset. The authors noted that performance was likely limited by computational constraints and the need for further hyperparameter fine-tuning.
* **Conclusion**: While SlateQ showed better accuracy in these trials, Double DQN exhibited more stability across different dataset sizes.

## Technical Setup
* **Optimization**: Adam optimizer was used for the neural network training.
* **Hyperparameters**: Best results for SlateQ were found with a learning rate of 0.00001, 128 hidden layers, and a slate size of 7.
* **Environment**: Experiments were primarily conducted locally on a CPU-based system (1.2 GHz Intel Core i7 with 16GB RAM).
