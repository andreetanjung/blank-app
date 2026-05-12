# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

---

## Deploying the app

The easiest option is Streamlit Community Cloud.

1. Push your repo to GitHub.
2. Go to https://streamlit.io/cloud and connect your GitHub account.
3. Select this repository and deploy the `main` branch.
4. Streamlit Cloud will install packages from `requirements.txt` and launch `streamlit_app.py`.

If your app uses the Anthropic key, add it under Settings > Secrets in Streamlit Cloud and reference it in the app securely.