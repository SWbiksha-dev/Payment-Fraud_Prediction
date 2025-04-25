import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import shap
import matplotlib.pyplot as plt
import requests

# -----------------------------
# Load the fraud detection model
model = joblib.load('pay.pkl')
model_features = ['step', 'type', 'amount']
type_mapping = {'CASH_OUT': 1, 'TRANSFER': 4, 'CASH_IN': 5, 'DEBIT': 2, 'PAYMENT': 3}
threshold = 0.85
# -----------------------------

st.set_page_config(page_title="💳 Fraud Detection & Chatbot", layout="centered")
st.title("💳 Real-Time Fraud Detection System")

tab1, tab2, tab3 = st.tabs(["📋 Manual Entry", "📁 CSV Upload", "💬 Chatbot Support"])

# -----------------------------
# 📋 Tab 1: Manual Entry
# -----------------------------
with tab1:
    st.subheader("Enter Transaction Details")
    step = st.number_input("Step (Hour)", min_value=0, max_value=744)
    t_type = st.selectbox("Transaction Type", list(type_mapping.keys()))
    amount = st.number_input("Transaction Amount ($)", min_value=0.0)

    if st.button("🔍 Predict Fraud"):
        input_data = pd.DataFrame([{
            'step': step,
            'type': type_mapping[t_type],
            'amount': amount
        }])

        score = model.predict_proba(input_data)[0][1]
        prediction = "🚨 Fraud" if score >= threshold else "✅ Not Fraud"

        st.metric("Fraud Score", f"{score:.4f}")
        st.success(f"Prediction: {prediction}")

        # SHAP Explanation
        st.subheader("📊 SHAP Explanation (Why this prediction?)")
        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)

        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

        st.subheader("🔍 Feature Contributions")
        for i, feature in enumerate(model_features):
            contribution = shap_values.values[0][i]
            st.write(f"🔹 **{feature.capitalize()}** contributed **{contribution:.4f}** to the fraud decision.")

# -----------------------------
# 📁 Tab 2: CSV Upload
# -----------------------------
with tab2:
    st.subheader("Upload CSV File")
    file = st.file_uploader("Upload your transaction file", type=["csv"])

    if file:
        df = pd.read_csv(file)

        try:
            df['type'] = df['type'].map(type_mapping)
            df_input = df[model_features]

            df['fraud_score'] = model.predict_proba(df_input)[:, 1]
            df['prediction'] = np.where(df['fraud_score'] >= threshold, '🚨 Fraud', '✅ Not Fraud')

            st.write("📊 Predictions Preview")
            st.dataframe(df[['step', 'type', 'amount', 'fraud_score', 'prediction']])

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", csv, "fraud_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# -----------------------------
# 💬 Tab 3: AI Chatbot + FAQ
# -----------------------------
faq_answers = {
    "Why was my transaction flagged as fraud?":
        "Your transaction had characteristics that matched known fraud patterns. Please check the details and confirm if it was you who made it.",
    "What should I do if this wasn't me?":
        "If you didn’t authorize the transaction, report it immediately through this chatbot or contact customer support.",
    "Can I confirm that this transaction is safe?":
        "Yes! Just let us know if the transaction was made by you, and we’ll mark it as safe in our system.",
    "Will my account be locked?":
        "Not immediately. We flag suspicious activity for review, and repeated suspicious activity may lead to a temporary hold for your safety.",
    "How can I prevent future fraud?":
        "Avoid sharing account details, use strong passwords, enable two-factor authentication, and monitor your account regularly.",
    "How did you decide this might be fraud?":
        "We use AI to analyze transaction type, amount, and behavior patterns. When something unusual happens, we alert you just in case.",
    "Can I speak to a human support agent?":
        "Yes. You can escalate this chat to our support team or call customer care for help at any time.",
    "Can I see the reason for this alert?":
        "Yes! You can view the breakdown of why this transaction was flagged under the 'Explanation' tab in your dashboard.",
    "Is my money safe now?":
        "Yes. If the transaction is still pending, we will hold it temporarily until you confirm whether it's genuine or not.",
    "What does the 'Fraud Score' mean?":
        "It’s a risk level from 0 to 1. A score closer to 1 means it’s more likely to be fraud. But don’t worry — we ask you before taking any action!"
}

with tab3:
    st.subheader("💬 Fraud Detection Support Bot")

    st.markdown("📌 **Select a common question or ask your own below**")
    selected_faq = st.selectbox("🤔 Choose a FAQ:", [""] + list(faq_answers.keys()))

    if selected_faq and selected_faq in faq_answers:
        st.markdown(f"**🤖 Bot:** {faq_answers[selected_faq]}")

    user_input = st.text_input("🗣️ Ask something:")
    if st.button("💬 Ask Bot"):
        if user_input.strip() == "":
            st.warning("Please enter a question.")
        else:
            try:
                # Send to TinyLlama via Ollama local API
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "tinyllama", "prompt": user_input, "stream": False}
                )
                bot_reply = response.json()["response"]
                st.markdown(f"**🤖 Bot:** {bot_reply}")
            except Exception as e:
                st.error(f"⚠️ Ollama Chat Error: {e}")
