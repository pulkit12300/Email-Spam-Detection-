# Email-Spam-Detection-
---

This project is designed to detect whether an email message is **spam** or **ham (not spam)** using a machine learning model built with Python. The dataset used in this project contains labeled email messages and is read from a tab-delimited CSV file. The model helps automate the identification of spam messages based on textual patterns in the content.

The project begins by importing the necessary Python libraries such as `pandas` for data handling, `numpy` for numerical operations, and `scikit-learn` modules for machine learning tasks. The dataset is loaded using `pandas.read_csv()` with proper encoding and tab delimiter settings. Since the raw dataset doesn’t include column headers, the script manually assigns the column names as `'label'` for the classification (spam/ham) and `'message'` for the email content.

To prepare the data for model training, a text preprocessing function called `clean_text()` is defined. This function converts the messages to lowercase and can be extended to remove punctuation, stopwords, or numbers. The cleaned messages are stored in a new column called `cleaned_message`. The labels ('ham' and 'spam') are also mapped to numeric values (`0` for ham, `1` for spam`) to make them suitable for machine learning.

Next, the textual data is vectorized using the **TF-IDF (Term Frequency-Inverse Document Frequency)** approach. This converts the text into numerical features that represent how important each word is to a message. The data is then split into training and testing sets using an 80-20 ratio with `train_test_split`.

A **Multinomial Naive Bayes** model is trained on the training data, which is well-suited for text classification problems. After training, the model is evaluated using metrics such as **accuracy**, **classification report**, and **confusion matrix** to understand its performance on the entire dataset.

The project also includes a function called `predict_email()` that takes an email text as input, processes and vectorizes it, and uses the trained model to predict whether the message is spam or not. A couple of example predictions are provided to demonstrate how the function works on new messages.

This project is a complete pipeline for building a basic spam detection system using machine learning, and can be further enhanced by improving text preprocessing, experimenting with other classifiers, or deploying the model using a simple web interface.

Let me know if you'd also like the `requirements.txt`, `.py` file structure, or deployment guide!
