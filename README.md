🥑 Avocado Price Prediction
===============================

Overview
--------
This project predicts avocado prices using machine learning models. Through extensive exploratory data analysis (EDA) and feature engineering, I developed and compared Support Vector Regression (SVR) and Random Forest models to identify the most effective approach for price prediction.

Dataset
-------
The dataset contains historical data on avocado prices including features such as:
- Region
- Type (conventional/organic)
- Volume information
- Date-related features

Methodology
-----------

Data Preprocessing:
- Performed thorough data cleaning and validation
- Extracted temporal features from dates (month, year, day of week)
- Applied one-hot encoding to categorical variables (region, type)
- Implemented feature scaling using StandardScaler

Feature Engineering:
- Created meaningful date-based features
- Properly encoded categorical variables to avoid data leakage
- Removed unnecessary columns to improve model performance

Model Development
----------------
I implemented and compared two regression models:

1. Support Vector Regression (SVR)
   - Used RBF kernel
   - Applied proper sparse matrix handling with ColumnTransformer
   - Achieved ~0.80 R² score on test data

2. Random Forest Regression
   - Implemented with optimal parameters
   - Leveraged ensemble learning capabilities
   - Achieved ~0.92 R² score on test data

Results
-------
| Model         | R² Score | Key Characteristics                      |
|---------------|----------|------------------------------------------|
| SVR           | 0.80     | Good generalization, slower training     |
| Random Forest | 0.92     | Superior performance, feature importance |

Key Learnings
------------
- Proper categorical encoding significantly impacts model performance
- Feature engineering is crucial for extracting meaningful patterns
- Random Forest outperformed SVR for this particular dataset
- Avoiding data leakage is essential for creating reliable models

Technical Implementation
-----------------------
- Used scikit-learn's ColumnTransformer for preprocessing pipeline
- Implemented proper train-test splitting
- Applied appropriate feature scaling
- Used sparse matrix handling techniques for efficient processing

How to Run
----------
# Install dependencies
pip install pandas numpy scikit-learn

# Run SVR model
python main.py

# Run Random Forest model
python main_rf.py

Project Structure
----------------
regression-avocado/
├── avocado1.csv           # The dataset
├── main.py                # SVR model with ColumnTransformer + OneHotEncoder
├── main_rf.py             # Random Forest model with the same pipeline
└── README.txt             # This file

Conclusion
----------
This project demonstrates my ability to:
- Perform meaningful EDA
- Apply proper ML preprocessing techniques
- Implement and compare regression models
- Achieve high predictive performance (R² of 0.92)

The Random Forest model proved most effective for avocado price prediction, likely due to its ability to capture non-linear relationships and handle the diverse regional factors affecting prices.

