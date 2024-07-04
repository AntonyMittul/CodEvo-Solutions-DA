import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



data = {
    "Review ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    "Rating": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 2, 1, 1],
    "Sentiment": ["Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Neutral", "Neutral", "Positive", "Positive", "Positive", "Positive", "Negative", "Negative", "Negative", "Negative", "Negative", "Negative"]
}

df = pd.DataFrame(data)

# Bar Chart 
plt.figure(figsize=(10, 6))
sns.countplot(x='Rating', data=df, palette='viridis')
plt.title('Ratings Distribution')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.show()

# Pie Chart 
sentiment_counts = df['Sentiment'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Sentiment Distribution')
plt.show()



# Calculating the Average Rating
average_rating = df['Rating'].mean()
print(f'Average Rating: {average_rating:.2f}')





