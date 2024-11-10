import pandas as pd 
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib

#dataset
url = "insurance.csv"
insurance_data = pd.read_csv(url)


# label encoding for converting categorical to numerical

label_encoder = LabelEncoder()
insurance_data['sex'] = label_encoder.fit_transform(insurance_data['sex'])

insurance_data['smoker'] = label_encoder.fit_transform(insurance_data['smoker'])

insurance_data['region'] = label_encoder.fit_transform(insurance_data['region'])


# features and target variables (y)

x= insurance_data.drop('expenses',axis=1)
y= insurance_data['expenses']

#  data spliting into test and train

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#  Model training

model= LinearRegression()
model.fit(X_train,y_train)

#  prediction on test set
y_pred = model.predict(X_test)


# model evaluation

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:",mse)

joblib.dump(model, 'insurance_model.pk1')



# ************************
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sqlite3

# Load the trained model
model = joblib.load('insurance_model.pk1')

# Create a label encoder for converting categorical variables
label_encoder = LabelEncoder()
label_encoder.fit(['male', 'female', 'yes', 'no', 'northeast', 'northwest', 'southeast', 'southwest'])

# Define entry widgets globally
username_entry = None
password_entry = None
signup_username_entry = None
signup_password_entry = None
age_entry = None
bmi_entry = None
children_entry = None
sex_var = None
smoker_var = None
region_var = None

# Function to handle login button click
def login():
    username = username_entry.get()
    password = password_entry.get()

    # Basic authentication (replace with secure authentication in a real application)
    if authenticate_user(username, password):
        show_dashboard()
    else:
        login_status.config(text='Invalid credentials. Would you like to sign up?', foreground="red")
        signup_prompt.pack()

# Function to handle signup button click
def signup():
    username = signup_username_entry.get()
    password = signup_password_entry.get()

    # Save the user to the database
    save_user(username, password)
    signup_status.config(text='User created successfully. You can now log in.', foreground="green")
    signup_prompt.pack_forget()

# Function to authenticate the user against the database
def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Create the users table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    # Check if the user exists in the database
    cursor.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
    user = cursor.fetchone()

    conn.close()

    return user is not None

# Function to save user to the database
def save_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Create the users table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    # Insert the new user
    cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

# Function to show the dashboard after successful login
def show_dashboard():
    login_frame.pack_forget()
    signup_frame.pack_forget()
    dashboard_frame.pack()

# Function to go back to login screen after signup
def back_to_login():
    login_frame.pack()
    signup_frame.pack_forget()
    signup_status.config(text='', foreground="green")
    signup_prompt.pack_forget()

# Function to predict expenses and display result
def predict_expenses():
    # Get input values from the user
    age = int(age_entry.get())
    bmi = float(bmi_entry.get())
    children = int(children_entry.get())
    sex = sex_var.get()
    smoker = smoker_var.get()
    region = region_var.get()

    # Convert categorical variables to numerical using Label Encoding
    sex = label_encoder.transform([sex])[0]
    smoker = label_encoder.transform([smoker])[0]
    region = label_encoder.transform([region])[0]

    # Use the same columns used during training
    custom_input = pd.DataFrame([[age, bmi, children, sex, smoker, region]],
                                 columns=x.columns)

    # Make a prediction
    prediction = model.predict(custom_input)

    # Display the prediction
    result_label.config(text=f'Predicted Insurance Expenses: ${prediction[0]:.2f}')

    # Debugging: Print feature names during training and in custom_input
    print("Feature Names during Training:", model.feature_names_in_)
    print("Feature Names in custom_input:", custom_input.columns)

    # Make a prediction
    prediction = model.predict(custom_input)

    # Display the prediction
    result_label.config(text=f'Predicted Insurance Expenses: ${prediction[0]:.2f}')

# Create the main application window using ThemedTk
app = ThemedTk(theme="arc")  # Set the theme ("arc" is just an example, you can choose another theme)
app.title('Insurance Expenses Prediction')
app.geometry("400x300")  # Set the initial size of the window

# Login Frame
login_frame = ttk.Frame(app, padding="20")
login_frame.pack(expand=True, fill="both")

login_label = ttk.Label(login_frame, text='Login', font=("Helvetica", 16))
login_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

username_label = ttk.Label(login_frame, text='Username:')
username_label.grid(column=0, row=1, pady=5, sticky="e")
username_entry = ttk.Entry(login_frame)
username_entry.grid(column=1, row=1, pady=5, sticky="w")

password_label = ttk.Label(login_frame, text='Password:')
password_label.grid(column=0, row=2, pady=5, sticky="e")
password_entry = ttk.Entry(login_frame, show='*')
password_entry.grid(column=1, row=2, pady=5, sticky="w")

login_button = ttk.Button(login_frame, text='Login', command=login)
login_button.grid(column=0, row=3, columnspan=2, pady=10)

signup_prompt = ttk.Label(login_frame, text='New user? Click here to sign up.', foreground="blue", cursor="hand2")
signup_prompt.grid(column=0, row=4, columnspan=2, pady=5)
signup_prompt.bind("<Button-1>", lambda event: show_frame(signup_frame))

login_status = ttk.Label(login_frame, text='', foreground="red")
login_status.grid(column=0, row=5, columnspan=2, pady=(10, 0))

# Signup Frame
signup_frame = ttk.Frame(app, padding="20")
signup_frame.pack(expand=True, fill="both")

signup_label = ttk.Label(signup_frame, text='Sign Up', font=("Helvetica", 16))
signup_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

signup_username_label = ttk.Label(signup_frame, text='Username:')
signup_username_label.grid(column=0, row=1, pady=5, sticky="e")
signup_username_entry = ttk.Entry(signup_frame)
signup_username_entry.grid(column=1, row=1, pady=5, sticky="w")

signup_password_label = ttk.Label(signup_frame, text='Password:')
signup_password_label.grid(column=0, row=2, pady=5, sticky="e")
signup_password_entry = ttk.Entry(signup_frame, show='*')
signup_password_entry.grid(column=1, row=2, pady=5, sticky="w")

signup_button = ttk.Button(signup_frame, text='Sign Up', command=signup)
signup_button.grid(column=0, row=3, columnspan=2, pady=10)

back_to_login_button = ttk.Button(signup_frame, text='Back to Login', command=back_to_login)
signup_status = ttk.Label(signup_frame, text='', foreground="green")
signup_status.grid(column=0, row=4, columnspan=2, pady=(10, 0))

# Hide signup frame initially
signup_frame.pack_forget()

# Dashboard Frame
dashboard_frame = ttk.Frame(app, padding="20")

dashboard_label = ttk.Label(dashboard_frame, text='Insurance Expense Prediction', font=("Helvetica", 16))
dashboard_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

age_label = ttk.Label(dashboard_frame, text='Age:')
age_label.grid(column=0, row=1, pady=5, sticky="e")
age_entry = ttk.Entry(dashboard_frame)
age_entry.grid(column=1, row=1, pady=5, sticky="w")

bmi_label = ttk.Label(dashboard_frame, text='BMI:')
bmi_label.grid(column=0, row=2, pady=5, sticky="e")
bmi_entry = ttk.Entry(dashboard_frame)
bmi_entry.grid(column=1, row=2, pady=5, sticky="w")

children_label = ttk.Label(dashboard_frame, text='Children:')
children_label.grid(column=0, row=3, pady=5, sticky="e")
children_entry = ttk.Entry(dashboard_frame)
children_entry.grid(column=1, row=3, pady=5, sticky="w")

sex_label = ttk.Label(dashboard_frame, text='Sex:')
sex_label.grid(column=0, row=4, pady=5, sticky="e")
sex_var = tk.StringVar()
sex_combobox = ttk.Combobox(dashboard_frame, textvariable=sex_var, values=['male', 'female'])
sex_combobox.grid(column=1, row=4, pady=5, sticky="w")
sex_combobox.set('male')

smoker_label = ttk.Label(dashboard_frame, text='Smoker:')
smoker_label.grid(column=0, row=5, pady=5, sticky="e")
smoker_var = tk.StringVar()
smoker_combobox = ttk.Combobox(dashboard_frame, textvariable=smoker_var, values=['yes', 'no'])
smoker_combobox.grid(column=1, row=5, pady=5, sticky="w")
smoker_combobox.set('no')

region_label = ttk.Label(dashboard_frame, text='Region:')
region_label.grid(column=0, row=6, pady=5, sticky="e")
region_var = tk.StringVar()
region_combobox = ttk.Combobox(dashboard_frame, textvariable=region_var, values=['northeast', 'northwest', 'southeast', 'southwest'])
region_combobox.grid(column=1, row=6, pady=5, sticky="w")
region_combobox.set('northeast')

predict_button = ttk.Button(dashboard_frame, text='Predict', command=predict_expenses)
predict_button.grid(column=0, row=7, columnspan=2, pady=10)

result_label = ttk.Label(dashboard_frame, text='', font=("Helvetica", 12))
result_label.grid(column=0, row=8, columnspan=2, pady=10)

# Hide dashboard frame initially
dashboard_frame.pack_forget()

def show_frame(frame):
    login_frame.pack_forget()
    signup_frame.pack_forget()
    dashboard_frame.pack_forget()
    frame.pack(expand=True, fill="both")

app.mainloop()
