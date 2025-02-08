import pandas as pd
import math
from sklearn import linear_model
from word2number import w2n

df = pd.read_csv("data.csv", index_col=False)

df.experience = df.experience.fillna("zero")

df["text_score(out of 10)"] = df["text_score(out of 10)"].fillna(
    math.floor(df["text_score(out of 10)"].median())
)

experience = df.experience.to_numpy()

for i in range(0, len(experience)):
    experience[i] = w2n.word_to_num(experience[i])

df.experience = pd.DataFrame(experience)

features = [
    "experience",
    "text_score(out of 10)",
    "interview_score(out of 10)",
]
X = df[features].values

y = df["salary($)"].values

model = linear_model.LinearRegression()
model.fit(X, y)

exp = int(input("Enter user experience (in years): "))
t_s = int(input("Enter test score (out of 10): "))
i_s = int(input("Enter the interview score (out of 10): "))

prediction = model.predict([[exp, t_s, i_s]])

print(f"The salary you should provide this interviewee is {prediction[0]}!")
