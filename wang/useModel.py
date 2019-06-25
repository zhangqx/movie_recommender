import os

import tensorflow as tf
import numpy as np
import random
import pickle
from flask import Flask
from flask import request
import json
from flask_cors import CORS

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

app = Flask(__name__)
CORS(app, supports_credentials=True)
movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))
users_matrics = pickle.load(open('users_matrics.p', mode='rb'))

# 从本地读取数据
title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(open('preprocess.p', mode='rb'))
#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]:i for i, val in enumerate(movies.values)}


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))


load_dir = load_params()
print('load_dir: ', load_dir)


def recommend_same_type_movie(movie_id_val, top_k=20):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        print('load_dir: ', load_dir)
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keep_dims=True))
        normalized_movie_matrics = movie_matrics / norm_movie_matrics

        # 推荐同类型的电影
        probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
        sim = (probs_similarity.eval())
        #     results = (-sim[0]).argsort()[0:top_k]
        #     print(results)

        print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))
        print("以下是给您的推荐：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        movies = []
        while len(results) != top_k:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
            print(val)
            print(movies_orig[val])
            movie = {}
            movie['id'] = movies_orig[val][0]
            movie['name'] = movies_orig[val][1]
            movie['type'] = movies_orig[val][2]
            movies.append(movie)

        return movies


# recommend_same_type_movie(1401, 20)


# 推荐您喜欢的电影

def recommend_your_favorite_movie(user_id_val, top_k=10):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # 推荐您喜欢的电影
        probs_embeddings = (users_matrics[user_id_val - 1]).reshape([1, 200])

        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
        sim = (probs_similarity.eval())
        #     print(sim.shape)
        #     results = (-sim[0]).argsort()[0:top_k]
        #     print(results)

        #     sim_norm = probs_norm_similarity.eval()
        #     print((-sim_norm[0]).argsort()[0:top_k])

        print("以下是给您的推荐：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        movies = []
        while len(results) != top_k:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
            print(val)
            print(movies_orig[val])
            movie = {}
            movie['id'] = movies_orig[val][0]
            movie['name'] = movies_orig[val][1]
            movie['type'] = movies_orig[val][2]
            movies.append(movie)

        return movies


# recommend_your_favorite_movie(234, 10)


# 看过这个电影的人还喜欢看什么电影
def recommend_other_favorite_movie(movie_id_val, top_k=20):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        probs_movie_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(users_matrics))
        favorite_user_id = np.argsort(probs_user_favorite_similarity.eval())[0][-top_k:]
        #     print(normalized_users_matrics.eval().shape)
        #     print(probs_user_favorite_similarity.eval()[0][favorite_user_id])
        #     print(favorite_user_id.shape)

        user_movies = {}
        print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))

        print("喜欢看这个电影的人是：{}".format(users_orig[favorite_user_id - 1]))
        usersArray = users_orig[favorite_user_id - 1]
        users = []
        for item in usersArray:
            user = {}
            user['id'] = item[0]
            user['sex'] = item[1]
            user['age'] = item[2]
            user['occupation'] = item[3]
            users.append(user)

        # users = "{}".format(users_orig[favorite_user_id - 1])
        user_movies['users'] = users
        probs_users_embeddings = (users_matrics[favorite_user_id - 1]).reshape([-1, 200])
        probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movie_matrics))
        sim = (probs_similarity.eval())
        #     results = (-sim[0]).argsort()[0:top_k]
        #     print(results)

        #     print(sim.shape)
        #     print(np.argmax(sim, 1))
        p = np.argmax(sim, 1)
        print("喜欢看这个电影的人还喜欢看：")

        results = set()
        movies = []
        # while len(results) != 5:
        #     c = p[random.randrange(top_k)]
        #     results.add(c)

        for i in range(top_k):
            c = p[i]
            results.add(c)

        for val in (results):
            print(val)
            print(movies_orig[val])
            movie = {}
            movie['id'] = movies_orig[val][0]
            movie['name'] = movies_orig[val][1]
            movie['type'] = movies_orig[val][2]
            movies.append(movie)

        user_movies['movies'] = movies
        return user_movies


# recommend_other_favorite_movie(1401, 20)


@app.route("/")
def is_started():
    return "Started!"


# 根据电影推荐相同类型的电影
@app.route("/function1", methods=["GET", "POST"])
def function1():
    if request.method == "POST":
        movie_id = request.form.get("movieId")
        count = request.form.get("count")

    else:
        movie_id = request.args.get("movieId")
        count = request.args.get("count")
    print(movie_id, count)

    result = recommend_same_type_movie(int(movie_id), int(count))
    return json.dumps(result)


# 根据用户推荐相同类型的电影
@app.route("/function2", methods=["GET", "POST"])
def function2():
    if request.method == "POST":
        user_id = request.form.get("userId")
        count = request.form.get("count")

    else:
        user_id = request.args.get("userId")
        count = request.args.get("count")
    print(user_id, count)

    result = recommend_your_favorite_movie(int(user_id), int(count))
    return json.dumps(result)


# 看过这个电影的人还喜欢看什么电影
@app.route("/function3", methods=["GET", "POST"])
def function3():
    if request.method == "POST":
        movie_id = request.form.get("movieId")
        count = request.form.get("count")

    else:
        movie_id = request.args.get("movieId")
        count = request.args.get("count")
    print(movie_id, count)

    result = recommend_other_favorite_movie(int(movie_id), int(count))
    return json.dumps(result)


if __name__ == '__main__':
    print('server is start at 7000')
    app.run(host='0.0.0.0', port=7000)