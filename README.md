# Medmesh

## Problem to Solve

Atrial fibrillation, the most common abnormal heart rhythm, causes 1 in 4 strokes. Two-thirds of those types of strokes are preventable with relatively inexpensive drugs. More people, including older populations most prone to stroke risk, are starting to use wearable technology such as Fitbit or the Apple Watch, which can monitor heart rate and track the daily activity.

However, current healthcare infrastructure is inefficient in terms of effectively making use of the extensive personal medical data available through wearable electronic devices and use of the online environment for better communication between doctors and patients.

## How our application solves the problem

We developed a web application where patient can provide his personal information and get a custom-made recommendation through an automated system.

To accomplish this goal, we first developed a machine learning model which predicts the probability of stroke due to atrial fibrillation. With 24/7 monitoring in the Fitbit, our application collects this information and passes it in as an input, in addition to other personal data collected during the sign-in, to determine the probability of having a stroke.
Finally, we offer custom-made recommendations to the patient through use of an interactive chatbot on our webpage.

## Requirements

Python 3.6 +

## How to start

1. Install requirements

```
cd server
pip3 install -r requirements.txt
```

2. Run Server

```
python server.py
```

3. Sign up and Click the embedded button on the bottom right

4. Enjoy chatting with Medmesh AI Chatbot
