import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─────────────────────────────────────────────
# 1. Realistic student feedback responses
# ─────────────────────────────────────────────
responses = [
    # POSITIVE responses
    "The lecture was really interesting and well explained",
    "I understood everything the lecturer taught today",
    "Today's class was very engaging and I learned a lot",
    "The topic was clear and the examples were very helpful",
    "I enjoyed the lecture, it was easy to follow",
    "Everything was well structured and I had no confusion",
    "The lecturer explained the concept in a simple way I could grasp",
    "I feel confident about the topic after today's class",
    "The session was productive and I took useful notes",
    "I understood the topic very well, great lecture",
    "Today was one of the best lectures so far",
    "The teaching style made the topic easy to understand",
    "I had no challenges, the class was smooth and clear",
    "I really enjoyed today and I am looking forward to the next class",
    "The lecture was detailed and covered everything I needed",

    # NEGATIVE responses
    "I did not understand most of what was taught today",
    "The lecture was too fast and I could not keep up",
    "I am confused about the topic, it was not clearly explained",
    "I struggled to follow the lecture from the beginning",
    "The class was boring and I lost focus halfway through",
    "I had many challenges understanding the new concept",
    "The explanation was unclear and I need more examples",
    "I felt lost during the lecture and could not take proper notes",
    "The topic was too complex and the lecturer moved too quickly",
    "I did not enjoy today's class, I understood very little",
    "There were too many distractions and I could not concentrate",
    "The lecture was poorly organized and hard to follow",
    "I need the topic to be repeated because I did not grasp it",
    "I was frustrated because I could not understand the material",
    "The class was not helpful for me today",

    # NEUTRAL responses
    "The lecture was okay but some parts were unclear",
    "I understood some sections but struggled with others",
    "It was an average class, nothing too difficult or too easy",
    "The topic was covered but I need to review my notes",
    "I followed most of the lecture but got confused at the end",
    "Some concepts were clear but others need more explanation",
    "The class was fine but I would have preferred more examples",
    "I understood the basics but the advanced part was confusing",
    "It was an average session, I will study more on my own",
    "The lecture was decent but could have been more engaging",
    "I had a few challenges but managed to understand the key points",
    "The class was neither too good nor too bad for me",
    "I understood about half of what was taught today",
    "Some parts were interesting but others were hard to follow",
    "It was okay overall, I just need to revisit a few sections",
]

questions = [
    "How was today's lecture?",
    "Did you understand the topic?",
    "What challenges did you face?"
]

# Assign questions in rotation across all responses
assigned_questions = [questions[i % len(questions)] for i in range(len(responses))]

# ─────────────────────────────────────────────
# 2. Semi-automatic sentiment labeling (VADER)
# ─────────────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()

def label_sentiment(text):
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

labels = [label_sentiment(r) for r in responses]
scores = [round(analyzer.polarity_scores(r)['compound'], 4) for r in responses]

# ─────────────────────────────────────────────
# 3. Build and save the DataFrame
# ─────────────────────────────────────────────
df = pd.DataFrame({
    "student_id": [f"student{(i % 32) + 1}" for i in range(len(responses))],
    "question": assigned_questions,
    "response": responses,
    "compound_score": scores,
    "sentiment_label": labels
})

output_path = "data/sentiment_feedback/transcripts/sentiment_dataset.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Dataset saved to {output_path}")
print(f"\n📊 Label Distribution:\n{df['sentiment_label'].value_counts().to_string()}")
print(f"\n🔍 Preview:\n{df.head(6).to_string(index=False)}")