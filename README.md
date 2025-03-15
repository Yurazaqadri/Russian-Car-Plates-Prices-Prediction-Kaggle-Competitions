# Russian-Car-Plates-Prices-Prediction-Kaggle-Competitions
Predict the market price of Russian car plates using listing data from aggregators
🔹 1. Importing Libraries

Pandas & NumPy → For data handling & numerical computations. Matplotlib & Seaborn → For visualization. sklearn.model_selection → Splitting the dataset into training and testing sets. sklearn.preprocessing → Encoding categorical features & standardizing numerical values. sklearn.ensemble → Using RandomForestRegressor for regression tasks. sklearn.linear_model → Includes Linear Regression. sklearn.svm → SVR (Support Vector Regression) for price prediction. sklearn.naive_bayes → Using GaussianNB, though it’s typically for classification. sklearn.metrics → To evaluate models using RMSE and R² score. REGION_CODES → A dictionary mapping plate codes to regions.

Loading & Displaying the Dataset
display(df.head()) display(df.info()) Loads the dataset from "train.csv". Uses display(df.head()) to show the first 5 rows. Uses df.info() to check data types & missing values.

Cleaning the Data
df = clean_data(df) Removes duplicate rows using drop_duplicates(). Drops missing values using dropna(). Converts the 'date' column to datetime format for easier manipulation.

Mapping Region Codes
Extracts the last 2-3 digits from the plate column using regex (\d{2,3}). Maps these extracted digits to region names using the REGION_CODES dictionary. Stores the mapped region name in the new region column.

Feature Engineering
df = feature_engineering(df) Encodes the categorical ‘region’ column into numbers using LabelEncoder(). Extracts features from the plate number: plate_length → Number of characters in the plate. has_triple_digits → Whether the plate has a 3-digit number. has_repeated_letters → Whether the plate has repeating letters. Extracts features from the date column: year, month, day, weekday.

🔹 6. Defining Features & Target Variable X (Features) → Selected attributes for training the model. y (Target Variable) → The price column.

🔹 7. Splitting Data into Training & Test Sets 80% training data, 20% test data for model evaluation. Random state = 42 ensures consistent results across runs.

Standardizing the Features
Standardizes features (mean = 0, std = 1) to improve model performance.

🔹 9. Training & Evaluating Different Models Trains a model using .fit(). Makes predictions using .predict(). Evaluates performance using: RMSE (Root Mean Square Error) → Measures prediction error. R² Score → Measures how well the model explains variance.

🔹 10. Running Multiple Models Trains 4 models: Linear Regression Random Forest Regressor Support Vector Regressor (SVR) Naïve Bayes (GaussianNB) Tracks the best model based on the lowest RMSE.
