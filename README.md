
---
title: Medicine System
emoji: 💊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---

# Medicine System App

An AI-powered medicine reminder and interaction checker built with Streamlit.

## Features
- Add medicines to a daily schedule (sidebar form).
- View, edit, and delete schedule entries directly in the table.
- A "next dose" callout and an automatic same-time conflict warning.
- Ask an AI (via Hugging Face) about a medicine's potential side effects.
- Ask the AI to scan your whole schedule for known drug-drug interactions.
- Pick which AI model to use, and test the connection, from the sidebar.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get a Hugging Face token**
   - Create a free account at [huggingface.co](https://huggingface.co).
   - Go to **Settings → Access Tokens** ([direct link](https://huggingface.co/settings/tokens)) and click **Create new token**.
   - A **fine-grained** token with the *"Make calls to Inference Providers"* permission works well (a classic **Read** token also works).
   - Copy the token (it starts with `hf_...`).

3. **Add your key to a `.env` file** in the project folder:
   ```
   HF_TOKEN=hf_your_token_here
   ```
   (The old `HUGGINGFACE_API_KEY` variable name still works too, if you had one set.)

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## About the AI model
Hugging Face now serves chat models through **Inference Providers** — a router
that forwards each request to whichever partner (Together, Fireworks,
Cerebras, HF's own servers, etc.) currently hosts that model, rather than one
single always-free backend. Because of that:

- The app defaults to `Qwen/Qwen2.5-7B-Instruct`, a small, ungated model that's
  broadly available. You can switch models from the **AI Settings** section in
  the sidebar if one isn't working for you.
- Hugging Face gives every account a small free monthly credit allowance for
  Inference Providers. If a request fails with a "credits used up" message,
  wait for the next billing cycle, add a payment method at
  [huggingface.co/settings/billing](https://huggingface.co/settings/billing),
  or pick a smaller model.
- Use the **Test AI connection** button in the sidebar any time you want to
  confirm your token/model combo works before using the chat features.

## Disclaimer
This app provides general information only and is not a substitute for
professional medical advice. Always consult a doctor or pharmacist about your
medications.