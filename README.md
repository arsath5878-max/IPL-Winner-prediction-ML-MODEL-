# 🏏 IPL Match Winner Predictor

A Machine Learning web application that predicts the **winner of an IPL match** based on historical IPL match data.
The project uses a **Random Forest Classifier** to analyze match features like teams, toss decision, and venue to predict the match outcome.

---

# 🚀 Project Overview

This project is designed to predict the probable winner of an IPL match using machine learning.
Users can select match details such as:

* Team 1
* Team 2
* Toss Winner
* Toss Decision
* Match Venue

The model processes these inputs and predicts:

* 🏆 Predicted Match Winner
* 📊 Winning Probability of Both Teams

The application is built using **Python, Scikit-Learn, and Streamlit**.

---

# 🧠 Machine Learning Model

The project uses the **Random Forest Classifier**, an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy.

### Steps Used in the Model

1. Dataset Loading
2. Data Cleaning
3. Feature Selection
4. Label Encoding (Convert text → numbers)
5. Train-Test Split
6. Model Training
7. Accuracy Evaluation
8. Model Saving using Pickle

---

# 📂 Dataset

The dataset used contains historical IPL match data.

### Important Features Used

| Feature       | Description            |
| ------------- | ---------------------- |
| team1         | First team playing     |
| team2         | Second team playing    |
| toss_winner   | Team that won the toss |
| toss_decision | Bat or Field           |
| venue         | Match stadium          |
| winner        | Actual match winner    |

---

# 🖥️ Web Application

The web interface is built using **Streamlit**, allowing users to interact with the model easily.

### User Inputs

* Select Team 1
* Select Team 2
* Select Toss Winner
* Select Toss Decision
* Select Venue

### Output

* Predicted Winner
* Winning Probability for both teams

---

# 🛠️ Technologies Used

* Python
* Pandas
* Scikit-Learn
* Streamlit
* Pickle
* Machine Learning (Random Forest)

---

# 📁 Project Structure

```
IPL-Winner-Predictor
│
├── matches.csv
├── train_model.py
├── app.py
├── ipl_model.pkl
├── team_encoder.pkl
├── toss_encoder.pkl
├── venue_encoder.pkl
├── winner_encoder.pkl
└── README.md
```

---

# ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/yourusername/
```
