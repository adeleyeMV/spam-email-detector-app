import streamlit as st
import joblib
from text_preprocessor import TextPreprocessor

# Load the serialized objects
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
spam_classifier = joblib.load('soft_voting_classifier.joblib')

# Create an instance of the preprocessor
preprocessor = TextPreprocessor()

def main():
    st.markdown("<h1 style='color: #FF4B4B;'>Spam Email Detection App</h1>", unsafe_allow_html=True)
    #st.title("Spam Email Detection App")
    st.markdown("<h6 >Enter the email you got to classify it as spam or not spam.</h6>", unsafe_allow_html=True)
    #st.write("Enter the email you got to classify it as spam or not spam.")

    # Text input
    user_input = st.text_area("Paste text here:", height=250)

    if st.button("Classify"):
        if user_input:
            # Preprocess the input text
            processed_text = preprocessor.preprocess(user_input)

            # Transform the processed text using the vectorizer
            vectorized_text = tfidf_vectorizer.transform([processed_text])

            # Predict using the classifier
            prediction = spam_classifier.predict(vectorized_text)[0]

            # Display the result
            if prediction == 1:
                st.markdown('The email is classified as: <span style="color:red;"><strong>Spam</strong></span>', unsafe_allow_html=True)
            else:
                st.markdown('The email is classified as: <span style="color:green;"><strong>Not Spam</strong></span>', unsafe_allow_html=True)
        else:
            st.write("Please enter some text to classify.")

if __name__ == "__main__":
    main()
