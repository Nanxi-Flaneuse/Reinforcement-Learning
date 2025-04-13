# Import statements
import json
import pandas as pd
import spacy
import numpy as np
import gym
import tensorflow as tf
import copy
import random
import pylab
import os
import gzip
from urllib.request import urlopen
from collections import deque
from keras import layers, models
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense

'''
need to incorporate: name of the film, year, reviewer profession?
need to encode occupation and gender!
'''

# # Load JSON data
# data = []
# with gzip.open('AMAZON_FASHION.json.gz') as f:
#     for l in f:
#         data.append(json.loads(l.strip()))
# # Let's take a peek at the first row and the total number of rows
# print(len(data))
# print(data[0])

# # Create a DataFrame for easier data manipulation
# df = pd.DataFrame(data)
# df = df[['overall','verified','user_id','asin','style','reviewerName','reviewText', 'summary','reviewTime']]
# # Filter verified reviews with non-null overall ratings
# filtered_df = df[(df['verified'] == True) & (~df['overall'].isnull())]
# print('hello')
filtered_df = pd.read_csv('data/ratings_updated_1.csv')


#Create FashionProduct class for a product representation from reviews
class Movie() : pass
class Reviewer() : pass
# Group reviews by reviewers and select users with more than ten purchases
reviewers = {}
grouped_df_reviwerId = filtered_df.groupby('user_id')

# creating review information on each user/reviewer
for user_id, group in grouped_df_reviwerId:
    movies = group['movie_id'].unique()
    profession = group['occupation'].iloc[0] 
    age = group['age'].iloc[0] 
    zip = group['zip'].iloc[0] 
    gender = group['gender'].iloc[0] 
    # if len(movies) > 10:
    reviewer = Reviewer()
    reviewer.user_id = user_id
    reviewer.movies = movies
    reviewer.profession = profession
    reviewer.age = age
    reviewer.zip = zip
    reviewers[user_id] = reviewer
# print(reviewers)
print('user database established')

# # Filter dataset to include only reviewers with more than ten products
# filtered_df = filtered_df[(filtered_df['user_id'].isin(list(reviewers.keys())))]
# print(filtered_df.head())
# print(len(filtered_df))
# Group reviews by product ASIN, user_id, and reviewTime
# filtered_df['reviewTime'] = pd.to_datetime(filtered_df['timestamp'])
filtered_df.sort_values('timestamp')
# grouped_df = filtered_df.groupby(['movie_id', 'user_id', 'timestamp'], sort=False)
# print(grouped_df.head())
#load spacy for nlp related noun extraction, stopword removal and others
nlp = spacy.load('en_core_web_sm')

# extract nouns from review text
# def extract_nouns(doc):
#     return " ".join([token.text for token in doc if token.pos_ == "NOUN" or token.pos_ == "PROPN"])

# Initialize a dictionary to store product features as states
states = {}
movies = {}

# Iterate over each product
# for (movie_id, user_id, timestamp), group in filtered_df:
for row in filtered_df.itertuples(index=False):
    if (row.movie_id, row.user_id) in states: continue
    movie = Movie()
    movie.movie_id = row.movie_id
    movie.user_id = row.user_id
    movie.time = row.timestamp
    movie.title = row.title
    movie.year = str(row.year)
    movie.genre = row.genres.split('|')
    # print('-------------------------------')
    # print(movie.genre)
    # print(type(movie.genre))
    if row.movie_id not in movies:
        movies[row.movie_id] = movie
        p= movies[row.movie_id]
        p.reviewers = set()
        p.rating =[]
        

    movie.reviewers = movies[row.movie_id].reviewers



    # extract genre metadata from genre column

    # movies[movie_id].update(genres)
    # products[product_asin].colors.update(colors)

    # # extract other noun metadata from review text
    # reviews = group[group['reviewText'].notna()]['reviewText']
    # reviews = " ".join(reviews.apply(lambda x: " ".join([extract_nouns(chunk) for chunk in nlp(x).noun_chunks]).strip()).unique())
    # products[product_asin].reviews.update(reviews)
    #using rms instead of average for review ratings to give slightly higher weightage to good reviews
    rating = int(row.rating)
    movies[row.movie_id].rating.append(rating)
    movie.rating_avg = np.sqrt(np.mean( [r**2 for r in movies[row.movie_id].rating]))
    # sizes = " ".join(products[product_asin].sizes)
    # colors = " ".join(products[product_asin].colors)
    # reviews = " ".join(products[product_asin].reviews)
    # gen = " ".join(movies[movie_id].genre)
    movie.metadata= 'title'+movie.title + ' '+'year'+movie.year + " ".join([f"genre{g}" for g in movie.genre])#' '.join([movie.title, movie.year, movie.genres])#" ".join((gen).split())

    # add past product and reviewer's product metadata
    # we will take metatdata of last 2 reviewer only as large metadata causes memory issues
    # for reviewer in list(movie.reviewers)[-2:]:
    #     state = states[(row.movie_id, reviewer)]
    #     movie.metadata += " "+state.metadata

    # keep past reviewer list
    movies[row.movie_id].reviewers.add(row.user_id)

    states[(row.movie_id, row.user_id)] = movie

states_list= list(states.values())
# users = {}

print('state generated')

# Create states for users and enhance metadata with past products
for state in states_list:
    if state.user_id not in reviewers:
        reviewers[state.user_id] = Reviewer()
        reviewers[state.user_id].movies = set()
    for prod1 in list(reviewers[state.user_id].movies)[-2:]:
        state1 = states[(prod1, state.user_id)]
        state.metadata += state1.metadata

# Remove states with empty metadata
states_list = [s for s in states_list if s.metadata.strip() != '']
print('state and movies set up successfully')

### implementing the DQN algorithm
# Create Product Recommendation env
class RecommendationEnv(gym.Env):
    def __init__(self, states, states_dict, iterations = 10):
        self.states = states
        self.state = self.states[0]
        self.states_dict = states_dict
        self.iterations = iterations
        self.index = 0
        state.action = 0


    def step(self, actions):
        # Implement the transition logic based on the action
        reward= 0
        done= False
        user_id= self.state.user_id
        future_asins= [p for p in reviewers[user_id].movies if self.states_dict[(p,user_id)].time>self.state.time]
        matched_recommendations = False
        #predicted recommendations
        for i in actions:
          if self.states[i].movie_id in future_asins:
            self.action = i
            matched_recommendations = True
            break


        if matched_recommendations:
            #Higher reward as they are bought products for the user in future
            reward = 1
        else:
            self.action = actions[0]


        self.index += 1
        self.state = self.states[self.index]
        print(f"iteration :{self.index}")
        if (self.iterations == self.index): done = True

        return self.state, reward, done, {}




    def reset(self, iterations = 10):
        # Reset the state to the initial position
        self.state = self.states[0]
        self.iterations = iterations
        self.index = 0
        return self.state

# Create the custom environment
env = RecommendationEnv(states_list, states, 10)

# Implementation of DQN algorithm
class DQNAgent:
    def __init__(self, state_size, action_size, states):
        self.states = states
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        #Using RNN as it is recommended for text classification
        # Using the TextVectorization layer to normalize, split, and map strings
        # to integers.
        encoder = tf.keras.layers.TextVectorization(max_tokens=10000)
        metadatas = [product.metadata for product in self.states]
        ratings = [product.rating_avg for product in self.states]
        encoder.adapt(metadatas)

        model = tf.keras.Sequential([
            encoder,
            layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
            layers.Bidirectional(layers.LSTM(64,  return_sequences=True)),
            layers.Bidirectional(tf.keras.layers.LSTM(32)),
            layers.Dense(64, activation='relu')
        ])
        # Create an input layer for ratings
        ratings_input = tf.keras.layers.Input(shape=(1,), name='ratings_input')

        # Concatenate the output of the previous layers with ratings
        concatenated = layers.concatenate([model.output, ratings_input])

        # Add additional layers for your desired architecture
        dense_layer = layers.Dense(64, activation='relu')(concatenated)
         # One Q-value per action
        output_layer = layers.Dense(len(self.states), activation='linear')(dense_layer)

        # Create the final model with both metadata and ratings as inputs
        model = tf.keras.Model(inputs=[model.input, ratings_input], outputs=output_layer)
        # Summary of the model
        model.summary()

        # Compile the model
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get recommendations from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.sample(range(self.action_size),10)
        else:
            q_value = self.model.predict([np.array([state.metadata]), np.array([state.ratings])])
            return np.argpartition(q_value[0],-10)[-10:]

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input_metadata =[]
        update_input_ratings =[]
        update_target_metadata = []
        update_target_ratings = []
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input_metadata.append(np.array(mini_batch[i][0].metadata))
            update_input_ratings.append(np.array(mini_batch[i][0].ratings))
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target_metadata.append(np.array(mini_batch[i][3].metadata))
            update_target_ratings.append(np.array(mini_batch[i][3].ratings))
            done.append(mini_batch[i][4])

        target = self.model.predict([np.transpose(update_input_metadata),np.transpose(update_input_ratings)])
        target_val = self.target_model.predict([np.transpose(update_target_metadata),np.transpose(update_target_ratings)])

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * ( np.amax(target_val[i]))

        # and do the model fit!
        self.model.fit([np.transpose(update_input_metadata),np.transpose(update_input_ratings)], target, batch_size=self.batch_size,
                       epochs=1, verbose=1)

state_size = len(env.states)
# Every other product can be a recommendation
action_size = state_size
agent = DQNAgent(state_size, action_size, env.states)

def run():
    scores, episodes = [], []
    EPISODES = 25
    #cache already rewarded recommendations (optimization done based upon context and to improve the performance to a large extent)
    next_states = {}
    done_value = {}
    action_value = {}
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset(50)

        while not done:
            if (state.movie_id, state.user_id) in next_states:
                next_state = next_states[(state.movie_id, state.user_id)]
                reward = 1
                done = done_value[(state.movie_id, state.user_id)]
                action = action_value[(state.movie_id, state.user_id)]
                env.index += 1
            else:
            # get action for the current state and go one step in environment
                actions = agent.get_action(state)
                next_state, reward, done, info = env.step(actions)
                action = env.action
                if (reward == 1):
                    next_states[(state.movie_id, state.user_id)]= next_state
                    done_value[(state.movie_id, state.user_id)] = done
                    action_value[(state.movie_id, state.user_id)] = env.action

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # every time step do the training
            agent.train_model()
            score += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()


                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                print("episode:", e, "  score:", score, "  memory length:",
                    len(agent.memory), "  epsilon:", agent.epsilon)
        if (score > 48): break

if __name__ == '__main__':
    # run()
    print(states)
    print(movies)