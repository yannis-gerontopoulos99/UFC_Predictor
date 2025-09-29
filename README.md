# UFC Predictor End-To-End Project 🥊

This project predicts UFC fight winners using machine learning models trained on scraped data from the official UFC website.
It achieves a ~70% accuracy and F1 score, one of the highest in any repo to date.
The project covers the data scraping, database usage, data exploration, cleaning, training, evaluation and local hosting. 

## 📌 Features

* **Web Scraping**: Extracts fighter stats, event details, and new fighter entries directly from UFC.com.
* **Database Management**: Stores and updates data in a MySQL database with custom scripts for fighters, stats, and events.
* **Data Processing**: Cleans and filters data (fighters, events, and stats) using Jupyter notebooks.
* **Feature Engineering**: Normalize and generate new features including averages, exponential moving averages (EMA), and momentum-based features.
* **Machine Learning Models**: Trains multiple classifiers (Random Forest, XGBoost, Logistic Regression, SVM, PyTorch, etc.) with standardization, and feature selection.
* **Performance**: Achieves ~70% accuracy, precision, and recall on historical and real-world test cases.

## 🗂️ Project Structure

```
.
├── .env
├── .gitignore
├── DB_connections
│   ├── append_events_DB.py
│   ├── append_fighters_DB.py
│   ├── append_stats_DB.py
│   ├── create_events_table_DB.py
│   ├── create_fighter_table_DB.py
│   └── create_stats_table_DB.py
├── LICENSE
├── README.md
├── app
│   ├── app.py
│   ├── app_flask.py
│   ├── static
│   │   └── ufc_logo.png
│   └── templates
│       └── index.html
├── data
│   ├── events.csv
│   ├── fighters.csv
│   ├── notebooks
│   │   ├── df_processed.csv
│   │   ├── events_cleaned.csv
│   │   ├── features_difference.csv
│   │   ├── features_selected.csv
│   │   ├── merged_clean.csv
│   │   ├── stats_cleaned.csv
│   │   ├── temporal_features.csv
│   │   ├── temporal_features_clean.csv
│   │   └── top_features.csv
│   └── stats.csv
├── docs
│   ├── adaboost.md
│   ├── decision_tree.md
│   ├── ema.md
│   ├── gradient_boosting.md
│   ├── hyperparameter_tuning.md
│   ├── knn.md
│   ├── logistic_regression.md
│   ├── momentum.md
│   ├── naive_bayes.md
│   ├── neural_network.md
│   ├── random_forest.md
│   ├── sgd.md
│   ├── svm.md
│   └── xgboost.md
├── models
│   ├── PyTorch_state_dict.pth
│   ├── adaboostclassifier.pkl
│   ├── decisiontreeclassifier.pkl
│   ├── encoder_stance.pkl
│   ├── gaussiannb.pkl
│   ├── gradientboostingclassifier.pkl
│   ├── kneighborsclassifier.pkl
│   ├── knn_imputer_feature_engineering.pkl
│   ├── knn_imputer_stats.pkl
│   ├── logisticregression.pkl
│   ├── randomforestclassifier.pkl
│   ├── scaler.pkl
│   ├── sgdclassifier.pkl
│   ├── svc.pkl
│   └── xgbclassifier.pkl
├── notebooks
│   ├── 1_etl.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_feature_engineering.ipynb
│   ├── 4_feature_selection.ipynb
│   ├── 5_trainning.ipynb
│   └── 6_predicting.ipynb
├── requirements.txt
└── scrape_data
    ├── compare_update_fighters.py
    ├── get_events.py
    ├── get_fighters.py
    ├── get_stats.py
    ├── update_events.py
    ├── update_fighters.py
    └── update_stats.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yannis-gerontopoulos99/UFC_Predictor.git
cd UFC_Predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up `.env` file with your MySQL database credentials:
```
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=yourpassword
DB_NAME=ufc_db
```

## Usage

**Scrape new data:**
```bash
python scrape_data/get_events.py
python scrape_data/get_fighters.py
python scrape_data/get_stats.py
```

**Create database:**
```bash
python DB_connections/create_events_table_DB.py
python DB_connections/create_fighters_table_DB.py
python DB_connections/create_stats_table_DB.py
```

**Append database:**
```bash
python DB_connections/append_events_DB.py
python DB_connections/append_fighters_DB.py
python DB_connections/append_stats_DB.py
```

**Update database:**
```bash
python scrape_date/update_events.py
```

**Logic behind database updating** Each time the script `scrape_date/update_events.py` is run, the new events and fights are being added.
In these fights, any new fighter that is found is being appended to the fighter database and his stats also in the stats database.
If an old fighter is found, then his stats are being updated in the stats database.
It also utilizes name normalization and matching due to the different nature of the fighter names from each database.

**UFC Website** In general, the website is well maintained and updated every week after an event. 
After scrapping all available data and analyzing it. It is noticed that some older events have some missing values.
The biggest issue are the fighter stats, which are a bit neglected and many issues with full name and nickname normalization.

**Explore data and train models:** Open the Jupyter notebooks in the `notebooks/` folder in order (1 → 6).

**Evaluate models:** Models are stored in `models/` and can be loaded for prediction.

## 💻 App

**Run app:** Run and predict using your input for fighter names
```bash
python app/app.py
```

**Local hosting:** Host and predict using your input for fighter names
```bash
python app/app_flask.py
```
**Docs:** Documentation of the models and methodologies

## 📊 Results

* Achieved ~70% **accuracy**, **precision**, and **recall**.
* Tested against real-world fight data for validation.

## 🛠️ Tech Stack

* **Python**: Data scraping, processing, and ML
* **MySQL**: Relational database storage
* **Scrapy**: Scraping
* **Pandas / NumPy**: Data manipulation
* **Scikit-learn / XGBoost / PyTorch**: Machine learning models
* **Jupyter Notebooks**: Analysis and experimentation
* **Rapidfuzz**: Fuzzy match for normalization
* **Flask**: Hosting the app

## 📝 Future Improvements

* Scrape and use betting data
* Explore more models
* Explore and perform different feature engineering techniques
* Add a logger and more error logic
* Host app in cloud
* Cache data while scrapping
* Create a more interactive UI
* Handle fighters with 1 fight

## 🙏 Acknowledgements

* Thanks to [@mfourier](https://github.com/mfourier) who provided influence and some documentation files that helped shape this project.
* Special thanks to the open-source community for the tools and libraries that make this possible.

## ⚖️ Disclaimer

* This project scrapes publicly available data from **UFC.com** for **educational and research purposes only**.
* All data, trademarks, and content are the **property of UFC** and their respective owners.
* This project is **not affiliated with or endorsed by UFC** in any way.
* Please respect the official website's Terms of Service.

## 📜 License

This project is licensed under the terms of the MIT License.