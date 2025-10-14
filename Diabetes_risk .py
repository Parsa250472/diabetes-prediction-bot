import telebot
from telebot import types
import joblib
import pandas as pd


API_TOKEN = "your token here"

model = joblib.load(r'C:\Users\EXO\diabetes_model4.pkl')
scaler = joblib.load(r'C:\Users\EXO\scaler3.pkl')

bot = telebot.TeleBot(API_TOKEN)

# start button
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    start_button = types.KeyboardButton("🚀 شروع پیش‌بینی دیابت")
    markup.add(start_button)

    bot.reply_to(message, "سلام! به ربات پیش‌بینی دیابت خوش آمدید 😊.\n"
                          "روی دکمه زیر کلیک کن یا اطلاعات رو به شکل زیر وارد کن:\n"
                          "سن, فشار خون بالا (1 یا 0), بیماری قلبی (1 یا 0), شاخص توده بدنی, HbA1c, گلوکز خون\n"
                          "مثال: 45,1,0,27.5,5.7,120", reply_markup=markup)

# when user clicks start button
@bot.message_handler(func=lambda message: message.text == "🚀 شروع پیش‌بینی دیابت")
def start_prediction(message):
    bot.reply_to(message, "لطفاً اطلاعاتت رو به این شکل وارد کن:\n"
                          "سن, فشار خون بالا (1 یا 0), بیماری قلبی (1 یا 0), شاخص توده بدنی, HbA1c, گلوکز خون\n"
                          "مثال: 45,1,0,27.5,5.7,120")

# preprocess input
def preprocess_input(data):
    input_df = pd.DataFrame([data])
    expected_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    input_df = input_df[expected_cols]
    input_scaled = scaler.transform(input_df)
    return input_scaled

# model prediction
@bot.message_handler(func=lambda message: True)
def predict_diabetes(message):
    try:
        user_input = [x.strip() for x in message.text.split(",")]
        if len(user_input) != 6:                        
            raise ValueError("به ۶ مقدار جدا شده با کاما نیاز دارم! لطفاً مثل نمونه وارد کن:45,1,0,27.5,5.7,120")

        user_data = {
            "age": float(user_input[0]),
            "hypertension": int(user_input[1]),
            "heart_disease": int(user_input[2]),
            "bmi": float(user_input[3]),
            "HbA1c_level": float(user_input[4]),
            "blood_glucose_level": float(user_input[5])
        }

        processed_input = preprocess_input(user_data)
        prediction = model.predict(processed_input)

        if prediction[0] == 1:
            bot.reply_to(message, "ممنون که اطلاعات رو فرستادی! بر اساس داده‌هات، ممکنه در معرض خطر دیابت باشی. پیشنهاد می‌کنم با پزشک مشورت کنی.")
        else:
            bot.reply_to(message, "خبر خوب! به نظر نمیاد در خطر دیابت باشی. سبک زندگی سالمت رو ادامه بده!")

        bot.reply_to(message, "توجه: این پاسخ صرفاً آموزشی است و جایگزین مشاوره پزشکی نیست. لطفاً برای بررسی دقیق‌تر به پزشک مراجعه کن. ❤️")

    except ValueError as ve:
        bot.reply_to(message, f"⛔️ {str(ve)} لطفاً اطلاعات رو دوباره وارد کن.")
    except Exception as e:
        bot.reply_to(message, f"یه مشکلی پیش اومد: {str(e)}. مطمئن شو فرمت رو درست وارد کردی: 45,1,0,27.5,5.7,120")

bot.polling()




import pandas as pd


df = pd.read_csv(r"C:\Users\EXO\.cache\kagglehub\datasets\iammustafatz\diabetes-prediction-dataset\versions\1\diabetes_prediction_dataset.csv")


# rows where diabetes is 1
diabetes_positive = df[df['diabetes'] == 1]

# show the filtered rows
print(diabetes_positive)


diabetes_positive.to_csv('diabetes_positive.csv', index=False)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv(r"C:\Users\EXO\diabetes_positive.csv")
df.head(10)
import pandas as pd

# Load both datasets
original_df = pd.read_csv(r'C:\Users\EXO\.cache\kagglehub\datasets\iammustafatz\diabetes-prediction-dataset\versions\1\diabetes_prediction_dataset.csv')
diabetes_positive_df = pd.read_csv(r'C:\Users\EXO\diabetes_positive.csv')

# concat dataset
combined_df = pd.concat([original_df, diabetes_positive_df], ignore_index=True)

# Check for duplicates 
combined_df = combined_df.drop_duplicates()

# Save the combined dataset 
combined_df.to_csv('combined_dataset.csv', index=False)

# Display the first few rows
print(combined_df.head())


df = pd.read_csv(r"C:\Users\EXO\combined_dataset.csv")


# features and target
features = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]
X = df[features]
y = df["diabetes"]

# Encode categorical features
X = pd.get_dummies(X, columns=["gender", "smoking_history"])

# Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)





import joblib
joblib.dump(model, 'diabetes_model3.pkl')  # Saves the model to a .pkl file
joblib.dump(scaler, 'scaler2.pkl')

import matplotlib.pyplot as plt
import numpy as np

# Define the metrics values (replace with actual values from your model)
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [0.85, 0.82, 0.79, 0.80]  # Replace with  results

# Create bar plot
plt.figure(figsize=(9, 6))
bars = plt.bar(metrics, values, color=["#1f77b4", "#27af27", "#ff7f0e", "#d62728"], edgecolor='black')

# Add grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set labels and title
plt.ylim(0, 1)
plt.ylabel("Score", fontsize=12)
plt.title("Model Performance Metrics", fontsize=14, fontweight='bold')

# Display values on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.03, f"{height:.2f}",
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()