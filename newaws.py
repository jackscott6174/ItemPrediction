# CS230 Project Code
# All code was originally in a Jupyter Notebook.

import numpy as np
import requests
import json
from pandas import read_csv
import matplotlib.pyplot as plt
import sklearn.metrics
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l1

# Item data collection
item_data = json.loads(requests.get('http://www.dota2.com/jsfeed/itemdata').text)['itemdata'].values()
item_names = []
big_items = set() # set of item number with cost > 1300
item_ids = {} # maps item number in the game's code to an "item id"
item_id = 0
for item in item_data:
    if item['cost'] is not None and item['cost'] > 1300:
        item_names.append(item['dname'])
        big_items.add(item['id'])
        item_ids[item['id']] = item_id
        item_id += 1
n_items = len(item_ids)

# COLLECTING MATCH DATA

# purchase data
all_purchases = read_csv('purchase_log.csv').values
# select only the expensive purchases
purchases = [p for p in all_purchases if p[0] in big_items]
for p in purchases:
    p[0] = item_ids[p[0]]

# select only matches at least 15 minutes long
durations = read_csv('match.csv', usecols=[2], squeeze=True).values
match_is_valid = durations > 900
valid_matches = np.arange(50000)[match_is_valid]
n_matches = len(valid_matches)
match_to_valid = {} # match number to internal match id that we use
for i in range(n_matches):
    match_to_valid[valid_matches[i]] = i

# randomly sampled times that we examine for each match
n_cuts = 8
times = np.random.rand(n_cuts, n_matches) * durations[valid_matches]

# hero selections for each player
heroes = read_csv('players.csv', usecols=np.arange(4)).values
n_heroes = np.max(heroes[:, 2])

# skill ratings for each player 
ratings = read_csv('player_ratings.csv', usecols=[0,3]).values
# we use average rating if the skill rating is not available
avg_rating = 0
id_to_rating = {}
for player_id, rating in ratings:
    if player_id >= 0:
        id_to_rating[player_id] = rating
        avg_rating += rating
avg_rating /= len(id_to_rating)

# FORMING TRAINING EXAMPLES
# for each player:
#   multi hot vector for item purchases
#   one hot vector for hero selection
#   float for skill rating
# output is multi hot vector for future purchases

n_players = 10 # per match
player_info_size = n_items + n_heroes + 1
n_y = n_players * n_items
n_x = n_players * player_info_size + 1
m = n_matches * n_cuts
X = np.zeros((m, n_x))
Y = np.zeros((m, n_y))

# mapping the game's player slot numbers to a 0-9 range
slots = [0, 1, 2, 3, 4, 128, 129, 130, 131, 132]
slot_to_id = {}
for i in range(len(slots)):
    slot_to_id[slots[i]] = i

# don't actually go through one match at a time,
# we just add each purchase to the correct training examples.
for item, time, slot, match in purchases:
    if not match_is_valid[match]:
        continue
    match = match_to_valid[match]
    for c in range(n_cuts):
        if time <= times[c][match]:
            X[match * n_cuts + c][slot_to_id[slot] * player_info_size + item] = 1
        else:
            Y[match * n_cuts + c][slot_to_id[slot] * n_items + item] = 1

# adding the current elapsed time for each example
for i in range(n_matches):
    for c in range(n_cuts):
        X[i * n_cuts + c][-1] = times[c][i]

# adding hero selections
for match_id, account_id, hero_id, player_slot in heroes:
    if not match_is_valid[match_id]:
        continue
    match_id = match_to_valid[match_id]
    for c in range(n_cuts):
        X[match_id * n_cuts + c][slot_to_id[player_slot] * player_info_size + n_items + hero_id] = 1
        X[match_id * n_cuts + c][slot_to_id[player_slot] * player_info_size + n_items + n_heroes] = id_to_rating.get(account_id, avg_rating)

# splitting examples into sets
m_dev = (int)(0.05 * m)
m_test = (int)(0.05 * m)
m_train = m - m_dev - m_test
match_ids = np.arange(m).reshape(-1, n_cuts)
np.random.seed(0) # keep random split consistent
np.random.shuffle(match_ids)
match_ids = match_ids.flatten()
dev = match_ids[:m_dev]
test = match_ids[m_dev:-m_train]
train = match_ids[-m_train:]

# TESTING

# some custom metrics
def true_pos(y_true, y_pred, threshold):
    return np.sum((y_pred >= threshold) * y_true)

def false_pos(y_true, y_pred, threshold):
    return np.sum((y_pred >= threshold) * (1 - y_true))

def true_neg(y_true, y_pred, threshold):
    return np.sum((y_pred < threshold) * (1 - y_true))

def false_neg(y_true, y_pred, threshold):
    return np.sum((y_pred < threshold) * y_true)

def pos(y_true):
    return np.sum(y_true)

def neg(y_true):
    return np.sum(1 - y_true)

def precision(y_true, y_pred):
    tp = true_pos(y_true, y_pred, 0.5)
    fp = false_pos(y_true, y_pred, 0.5)
    return tp / (tp + fp)

def recall(y_true, y_pred):
    tp = true_pos(y_true, y_pred, 0.5)
    fn = false_neg(y_true, y_pred, 0.5)
    return tp / (tp + fn)

def roc_auc(y_true, y_pred):
    fpr = []
    tpr = []
    n = neg(y_true)
    p = pos(y_true)
    for t in range(-40, 40): # sample a wide range of thresholds
        threshold = 1 / (1 + np.exp(t / 4))
        fpr.append(false_pos(y_true, y_pred, threshold) / n)
        tpr.append(true_pos(y_true, y_pred, threshold) / p)
    # add one more threshold data point to make auc calculation accurate
    fpr.append(false_pos(y_true, y_pred, 0) / n)
    tpr.append(true_pos(y_true, y_pred, 0) / p)
    return fpr, tpr, sklearn.metrics.auc(fpr, tpr)

# train a model using the given layers and output information about it using the dev set
# this is the main function used for testing different models
def eval_model(layers, lr=0.001, epochs=300):
    model = Sequential(layers)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['acc'])
    train_hist = model.fit(X[train], Y[train], epochs=epochs, batch_size=1024)
    loss, acc = model.evaluate(X[dev], Y[dev])
    y_pred = model.predict(X[dev])
    return [model, train_hist, loss, acc, roc_auc(Y[dev], y_pred), precision(Y[dev], y_pred), recall(Y[dev], y_pred)]

layers = [
    Dense(1500, input_shape=(n_x,), kernel_regularizer=l1(0.0000005)),
    BatchNormalization(),
    LeakyReLU(0.1),
    Dropout(0.1),
    Dense(1000, kernel_regularizer=l1(0.0000005)),
    BatchNormalization(),
    LeakyReLU(0.1),
    Dense(n_y),
    Activation('sigmoid')
]

def plot_roc(fpr, tpr):
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr)
    plt.title('ROC curve for test data')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

# function for testing on a specific example

name_data = read_csv('hero_names.csv', usecols=[1,2]).values
hero_names = {}
for hero_id, hero_name in name_data:
    hero_names[hero_id] = hero_name

def parse_example(model, match, player, cut):
    example = match * n_cuts + cut
    player_info = X[example][player * player_info_size : (player + 1) * player_info_size]
    
    print("Hero:", hero_names[np.where(player_info[n_items:-1])[0][0]])
    
    time = (int)(times[cut][match])
    print("Game Time:", (str)(time // 60) + ':' + str(time % 60))
    
    items = player_info[:n_items]
    print("Current Items:", [item_names[item_id] for item_id in np.where(items)[0]])
    
    future_items = np.where(Y[example, player * n_items : (player + 1) * n_items])[0]
    print("Future Items:", [item_names[item_id] for item_id in future_items])
    print("{0:<29} Probabilities:".format("Predicted Future Items:"))
    pred_items = model.predict(X[example : example + 1])[0][player * n_items : (player + 1) * n_items]
    rank = 1
    for item_id in sorted(range(n_items), key=lambda item_id:pred_items[item_id])[:-21:-1]:
        bar = '\u2588' * (int)(pred_items[item_id] * 40)
        if item_id in future_items:
            print(("{0:>2} {1:<26} [{2:<40}]").format(rank, item_names[item_id].upper(), bar))
        else:
            print(("{0:>2} {1:<26} [{2:<40}]").format(rank, item_names[item_id].lower(), bar))
        rank += 1