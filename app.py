from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("Instagram.csv", encoding = 'latin1')
data = data.dropna()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/impressions')
def impressions():
    plt.figure(figsize=(10, 8))
    plt.style.use('fivethirtyeight')
    plt.title("Distribution of Impressions From Home")
    sns.distplot(data['From Home'])
    plt.savefig('static/images/impressions_home.png')
    
    plt.figure(figsize=(10, 8))
    plt.title("Distribution of Impressions From Hashtags")
    sns.distplot(data['From Hashtags'])
    plt.savefig('static/images/impressions_hashtags.png')
    
    plt.figure(figsize=(10, 8))
    plt.title("Distribution of Impressions From Explore")
    sns.distplot(data['From Explore'])
    plt.savefig('static/images/impressions_explore.png')
    
    home = data["From Home"].sum()
    hashtags = data["From Hashtags"].sum()
    explore = data["From Explore"].sum()
    other = data["From Other"].sum()

    labels = ['From Home','From Hashtags','From Explore','Other']
    values = [home, hashtags, explore, other]

    fig = px.pie(data, values=values, names=labels, 
                 title='Impressions on Instagram Posts From Various Sources', hole=0.5)
    fig.write_image("static/images/impressions_pie.png")
    
    return render_template('impressions.html')

@app.route('/wordcloud')
def wordcloud():
    text = " ".join(i for i in data.Caption)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    plt.style.use('classic')
    plt.figure( figsize=(12,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('static/images/wordcloud_caption.png')
    
    text = " ".join(i for i in data.Hashtags)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    plt.figure( figsize=(12,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('static/images/wordcloud_hashtags.png')
    
    return render_template('wordcloud.html')

@app.route('/correlation')
def correlation():
    corr = data.corr()
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.colorbar()
    plt.savefig('static/images/correlation.png')
    
    return render_template('correlation.html')

@app.route('/regression')
def regression():
    x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                       'Profile Visits', 'Follows']])
    y = np.array(data["Impressions"])
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=42)
    model = LinearRegression()
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    score = model.score(xtest, ytest)
    mse = mean_squared_error(ytest, ypred)
    
    features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
    prediction = model.predict(features)
    
    return render_template('regression.html', score=score, mse=mse, prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)