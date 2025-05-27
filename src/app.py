from utils import db_connect
engine = db_connect()

# your code here
from pickle import load
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Cargar el modelo, el vectorizador y la data desde el archivo .sav

knn_model = load(open(r"knn_neighbors-6_algorithm-brute_metric-cosine.sav", "rb"))

total_data = load(open(r"clean_data.csv", "rb"))
    
vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b', lowercase=True)
matrix = vectorizer.fit_transform(total_data['tags'])

def get_movie_recommendations(movie_title):
    movie_index = total_data[total_data["title"] == movie_title].index[0]
    distances, indices = knn_model.kneighbors(matrix[movie_index])
    similar_movies = [(total_data["title"][i], distances[0][j]) for j, i in enumerate(indices[0])]
    return similar_movies[1:]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    movie_title = request.form['movie-title']
    recommended_movies = get_movie_recommendations(movie_title)
    return render_template('recommendations.html', movie_title=movie_title, recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)