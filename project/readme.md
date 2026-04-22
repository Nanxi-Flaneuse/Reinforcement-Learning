[cite_start]Based on the project description provided in the document "Comparison of Reinforcement Learning Techniques for Movie Recommendations"[cite: 4, 6], here is a generated README summary for your git repository:

# Comparison of Reinforcement Learning Techniques for Movie Recommendations

## Authors
[cite_start]Nicole Falicov, Nanxi Liu, Angela Shen [cite: 6]

## Project Overview
[cite_start]This project explores the development and comparison of two reinforcement learning (RL) methods for generating successful personalized movie recommendations[cite: 6]. [cite_start]Unlike traditional classification approaches, RL techniques are utilized here to maximize long-term user engagement and satisfaction by handling dynamic environments effectively[cite: 6].

## Methods
The project implements and compares the following two RL architectures:

* [cite_start]**SlateQ**: A decomposition of value-based temporal-difference and Q-learning designed to make RL tractable for "slates" (lists of recommendations) rather than single items[cite: 6]. [cite_start]It uses a neural network to approximate Q-values for user-item pairs based on concatenated user and movie embeddings[cite: 6].
* [cite_start]**Double Q-Learning (Double DQN)**: An extension of the Deep Q-Network (DQN) that uses a target network and experience replay[cite: 6]. [cite_start]It employs a second set of weights to reduce overoptimistic value estimates, leading to more stable learning outcomes[cite: 6].

## Dataset
[cite_start]Experiments were conducted using the **MovieLens Dataset** (Harper & Konstan, 2015) in two scales[cite: 6]:
* [cite_start]**MovieLens-100k**: 100,000 ratings from 943 users on 1,682 movies[cite: 6].
* [cite_start]**MovieLens-1M**: 1,000,000 ratings from 6,000 users on 4,000 movies[cite: 6].
[cite_start]Data was split using an 80/20 train-test ratio based on chronological rating timestamps to assess future viewing predictions[cite: 6].

## Evaluation Metrics
The models were evaluated against random baselines using the following metrics:
* [cite_start]**Top-k Hit Rate**: Measures the percentage of users for whom at least one recommended item was watched in the future (k=4)[cite: 6].
* [cite_start]**Normalized Discounted Cumulative Gain (nDCG@4)**: Evaluates the accuracy of the recommendation rankings and predicted ratings[cite: 6].

## Key Results
* [cite_start]**SlateQ Performance**: SlateQ performed approximately three times better than the random baseline, achieving a top-4 hit rate of **18%** and an nDCG of **12%** on the 100k dataset[cite: 6]. [cite_start]Performance scaled down slightly on the 1M dataset to a 10% hit rate[cite: 6].
* [cite_start]**Double DQN Performance**: Results were lower than anticipated, with a peak top-4 accuracy of **3.4%** on the 100k dataset[cite: 6]. [cite_start]The authors noted that performance was likely limited by computational constraints and the need for further hyperparameter fine-tuning[cite: 6].
* [cite_start]**Conclusion**: While SlateQ showed better accuracy in these trials, Double DQN exhibited more stability across different dataset sizes[cite: 6].

## Technical Setup
* [cite_start]**Optimization**: Adam optimizer was used for the neural network training[cite: 6].
* [cite_start]**Hyperparameters**: Best results for SlateQ were found with a learning rate of 0.00001, 128 hidden layers, and a slate size of 7[cite: 6].
* [cite_start]**Environment**: Experiments were primarily conducted locally on a CPU-based system (1.2 GHz Intel Core i7 with 16GB RAM)[cite: 6].
