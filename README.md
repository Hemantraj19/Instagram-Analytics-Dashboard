# Instagram Analytics Dashboard

This is a Flask web application that analyzes Instagram data and provides insights through various visualizations and machine learning regression.

## Features

1. **Home**: The home page with a simple welcome message.

2. **Impressions**: Analyzes the distribution of impressions from different sources and displays a pie chart.

3. **Wordcloud**: Generates word clouds from captions and hashtags in Instagram posts.

4. **Correlation**: Displays a heatmap showing the correlation between different features in the dataset.

5. **Regression**: Performs linear regression on selected features to predict impressions and shows the model's performance metrics.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/instagram-analytics.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:

   ```bash
   python app.py
   ```

   Open a web browser and go to [http://localhost:5000](http://localhost:5000) to access the dashboard.

## Directory Structure

- `static/`: Contains static assets like images.
- `templates/`: HTML templates for different pages.
- `app.py`: The main Flask application file.
- `Instagram.csv`: The Instagram data file in CSV format.
- `requirements.txt`: List of Python dependencies.

## Dependencies

- Flask
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Plotly Express
- WordCloud
- Scikit-learn

## Author

Hemant Raj
