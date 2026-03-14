import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("matches.csv")

# Select useful columns
data = data[['team1','team2','toss_winner','toss_decision','venue','winner']]

# Remove missing values
data = data.dropna()

# Create encoders
team_encoder = LabelEncoder()
toss_encoder = LabelEncoder()
venue_encoder = LabelEncoder()
winner_encoder = LabelEncoder()

# Encode columns
data['team1'] = team_encoder.fit_transform(data['team1'])
data['team2'] = team_encoder.transform(data['team2'])
data['toss_winner'] = team_encoder.transform(data['toss_winner'])
data['toss_decision'] = toss_encoder.fit_transform(data['toss_decision'])
data['venue'] = venue_encoder.fit_transform(data['venue'])
data['winner'] = winner_encoder.fit_transform(data['winner'])

# Split input and output
X = data.drop("winner", axis=1)
y = data["winner"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier()
model.fit(X_train,y_train)

# Accuracy
pred = model.predict(X_test)
accuracy = accuracy_score(y_test,pred)

print("Model Accuracy:", accuracy)

# Save model and encoders
pickle.dump(model, open("ipl_model.pkl","wb"))
pickle.dump(team_encoder, open("team_encoder.pkl","wb"))
pickle.dump(toss_encoder, open("toss_encoder.pkl","wb"))
pickle.dump(venue_encoder, open("venue_encoder.pkl","wb"))
pickle.dump(winner_encoder, open("winner_encoder.pkl","wb"))

print("Model and encoders saved successfully!")



import streamlit as st
import pickle
import pandas as pd

# Load model and encoders
model = pickle.load(open("ipl_model.pkl","rb"))
team_encoder = pickle.load(open("team_encoder.pkl","rb"))
toss_encoder = pickle.load(open("toss_encoder.pkl","rb"))
venue_encoder = pickle.load(open("venue_encoder.pkl","rb"))
winner_encoder = pickle.load(open("winner_encoder.pkl","rb"))

teams = list(team_encoder.classes_)
venues = list(venue_encoder.classes_)

st.title("🏏 IPL Match Winner Predictor")

team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)

toss_winner = st.selectbox("Toss Winner", teams)

toss_decision = st.selectbox("Toss Decision", ["bat","field"])

venue = st.selectbox("Match Venue", venues)

if st.button("Predict Winner"):

    # Encode inputs
    team1_encoded = team_encoder.transform([team1])[0]
    team2_encoded = team_encoder.transform([team2])[0]
    toss_encoded = team_encoder.transform([toss_winner])[0]
    decision_encoded = toss_encoder.transform([toss_decision])[0]
    venue_encoded = venue_encoder.transform([venue])[0]

    input_data = pd.DataFrame({
        "team1":[team1_encoded],
        "team2":[team2_encoded],
        "toss_winner":[toss_encoded],
        "toss_decision":[decision_encoded],
        "venue":[venue_encoded]
    })

    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    winner = winner_encoder.inverse_transform(prediction)[0]

    # probability calculation
    team1_index = winner_encoder.transform([team1])[0]
    team2_index = winner_encoder.transform([team2])[0]

    team1_prob = probabilities[0][team1_index] * 100
    team2_prob = probabilities[0][team2_index] * 100

    st.subheader("Match Prediction")

    st.write(f"{team1} Winning Probability: {team1_prob:.2f}%")
    st.write(f"{team2} Winning Probability: {team2_prob:.2f}%")

    st.success(f"Predicted Winner: {winner}")