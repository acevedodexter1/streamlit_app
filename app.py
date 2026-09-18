import os
from datetime import datetime, time as dtime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError, GatedRepoError

# =========================================================
# Setup
# =========================================================
load_dotenv()

# Support either env var name so existing .env files keep working.
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")

# A short, curated list of instruct models that are commonly served through
# Hugging Face's "Inference Providers" router. Hugging Face regularly changes
# which provider hosts which model, so we let the user switch models from the
# sidebar instead of hard-coding a single one that might stop working later.
MODEL_OPTIONS = {
    "Qwen2.5-7B-Instruct (recommended, ungated)": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral-7B-Instruct-v0.3 (ungated)": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama-3.2-3B-Instruct (Meta, license must be accepted on HF)": "meta-llama/Llama-3.2-3B-Instruct",
    "Zephyr-7B-beta (original/legacy, may be unavailable)": "HuggingFaceH4/zephyr-7b-beta",
}

DATA_PATH = "medicine_schedule.csv"

st.set_page_config(page_title="Medicine System", layout="wide", page_icon="💊")
st.title("💊 Medicine Reminder & Interaction Checker")


@st.cache_resource(show_spinner=False)
def get_client(token: str) -> InferenceClient:
    """Create (and cache) the Hugging Face Inference client.

    provider="auto" lets Hugging Face route the request to whichever partner
    (Together, Fireworks, Cerebras, HF's own servers, etc.) currently hosts
    the requested model, instead of assuming a single provider.
    """
    return InferenceClient(token=token, provider="auto")


def ask_hf(prompt: str, system_prompt: str, model: str, max_tokens: int = 400) -> str:
    """Send a chat request to Hugging Face and return the reply text.

    Raises a RuntimeError with a human-readable message on failure so the
    calling code can just show it with st.error().
    """
    if not HF_TOKEN:
        raise RuntimeError(
            "No Hugging Face token found. Add HF_TOKEN=hf_xxx to a `.env` file "
            "in the project folder, then restart the app."
        )

    client = get_client(HF_TOKEN)
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except GatedRepoError:
        raise RuntimeError(
            f"'{model}' is a gated model. Visit its page on huggingface.co while "
            "logged in and accept the license, then try again — or pick a "
            "different model in the sidebar."
        )
    except HfHubHTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 402:
            raise RuntimeError(
                "Hugging Face says your free inference credits are used up for "
                "this billing period (HTTP 402). Try again later, add a payment "
                "method at huggingface.co/settings/billing, or pick a smaller "
                "model in the sidebar."
            )
        elif status == 404:
            raise RuntimeError(
                f"'{model}' isn't currently available through Hugging Face's "
                "Inference Providers (HTTP 404). Pick a different model in the "
                "sidebar."
            )
        elif status in (401, 403):
            raise RuntimeError(
                "Hugging Face rejected the request as unauthorized (HTTP "
                f"{status}). Double-check that HF_TOKEN in your `.env` file is "
                "correct and hasn't expired."
            )
        else:
            raise RuntimeError(f"Hugging Face API error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while contacting Hugging Face: {e}")


def parse_time_safe(value: str):
    """Best-effort parse of a time string into a datetime.time, or None."""
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(str(value).strip(), fmt).time()
        except (ValueError, TypeError):
            continue
    return None


# =========================================================
# Load data
# =========================================================
if "df" not in st.session_state:
    if os.path.exists(DATA_PATH):
        st.session_state.df = pd.read_csv(DATA_PATH)
    else:
        st.session_state.df = pd.DataFrame(columns=["Medicine", "Dosage", "Time", "Notes"])


def save_df():
    st.session_state.df.to_csv(DATA_PATH, index=False)


# =========================================================
# Sidebar: Add Medicine
# =========================================================
st.sidebar.header("Add New Medicine")
with st.sidebar.form("medicine_form", clear_on_submit=True):
    med_name = st.text_input("Medicine Name")
    dosage = st.text_input("Dosage (e.g., 500mg)")
    time_val = st.time_input("Time to take", value=dtime(8, 0))
    notes = st.text_input("Notes (optional)")
    submitted = st.form_submit_button("Add to Schedule")

    if submitted:
        if not med_name.strip() or not dosage.strip():
            st.sidebar.error("Please enter at least a medicine name and dosage.")
        else:
            new_row = pd.DataFrame([{
                "Medicine": med_name.strip(),
                "Dosage": dosage.strip(),
                "Time": time_val.strftime("%H:%M"),
                "Notes": notes.strip(),
            }])
            st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
            save_df()
            st.sidebar.success(f"Added {med_name} to your schedule!")

st.sidebar.divider()
st.sidebar.header("🤖 AI Settings")
if HF_TOKEN:
    st.sidebar.success("Hugging Face token found.")
else:
    st.sidebar.error("No Hugging Face token found (set HF_TOKEN in `.env`).")

model_label = st.sidebar.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
selected_model = MODEL_OPTIONS[model_label]

if st.sidebar.button("Test AI connection"):
    with st.sidebar:
        with st.spinner("Pinging Hugging Face..."):
            try:
                reply = ask_hf(
                    "Reply with the single word OK.",
                    "You are a connectivity test. Reply with only the word OK.",
                    selected_model,
                    max_tokens=5,
                )
                st.success(f"Connected! Model replied: {reply.strip()!r}")
            except RuntimeError as e:
                st.error(str(e))

# =========================================================
# Main Area: Schedule
# =========================================================
st.subheader("📅 Your Daily Schedule")

if st.session_state.df.empty:
    st.info("No medicines added yet. Use the sidebar to add one.")
else:
    # Sort by time-of-day for display/editing so the schedule reads top-to-bottom.
    display_df = st.session_state.df.copy()
    display_df["_sort"] = display_df["Time"].apply(
        lambda v: parse_time_safe(v) or dtime(23, 59)
    )
    display_df = display_df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    # --- "Next up" callout ---
    now_t = datetime.now().time()
    parsed_times = [(i, parse_time_safe(t)) for i, t in enumerate(display_df["Time"])]
    parsed_times = [(i, t) for i, t in parsed_times if t is not None]
    upcoming = [(i, t) for i, t in parsed_times if t >= now_t]
    next_idx = upcoming[0][0] if upcoming else (parsed_times[0][0] if parsed_times else None)
    if next_idx is not None:
        row = display_df.iloc[next_idx]
        when = "today" if upcoming else "tomorrow"
        st.info(f"⏰ **Next up ({when}):** {row['Medicine']} ({row['Dosage']}) at {row['Time']}")

    # --- Same-time conflict check (no AI needed, instant) ---
    dupe_times = display_df["Time"][display_df["Time"].duplicated(keep=False)]
    if not dupe_times.empty:
        conflict_groups = display_df[display_df["Time"].isin(dupe_times.unique())]
        for t, group in conflict_groups.groupby("Time"):
            meds = ", ".join(group["Medicine"])
            st.warning(f"🕒 Scheduled at the same time ({t}): {meds}")

    st.caption("Edit cells directly, or select a row and use the trash icon to delete it.")
    edited_df = st.data_editor(
        display_df,
        use_container_width=True,
        num_rows="dynamic",
        key="schedule_editor",
    )
    if not edited_df.equals(display_df):
        st.session_state.df = edited_df.reset_index(drop=True)
        save_df()
        st.rerun()

st.divider()

# =========================================================
# AI: Interaction Checker (new)
# =========================================================
st.subheader("⚠️ Check Interactions Across Your Schedule")
st.write(
    "Ask the AI to review every medicine currently in your schedule for "
    "well-known interactions or timing considerations."
)
if st.button("Check My Full Schedule for Interactions"):
    if st.session_state.df.empty:
        st.warning("Add some medicines to your schedule first.")
    else:
        med_list = "\n".join(
            f"- {r.Medicine} ({r.Dosage}) at {r.Time}"
            for r in st.session_state.df.itertuples()
        )
        with st.spinner("Checking with AI..."):
            try:
                reply = ask_hf(
                    prompt=(
                        "Here is my current medicine schedule:\n"
                        f"{med_list}\n\n"
                        "Are there any known drug-drug interactions or timing "
                        "conflicts (e.g., medicines that should be spaced apart) "
                        "among these? Summarize with short bullet points."
                    ),
                    system_prompt=(
                        "You are a helpful medical-information assistant. Give "
                        "brief, clear, bullet-pointed information about possible "
                        "drug interactions and timing considerations. You are not "
                        "a doctor or pharmacist; always end by reminding the user "
                        "to confirm with one before making any changes."
                    ),
                    model=selected_model,
                    max_tokens=500,
                )
                st.markdown(reply)
            except RuntimeError as e:
                st.error(str(e))

st.divider()

# =========================================================
# AI: Side Effects Checker
# =========================================================
st.subheader("🤖 Ask About Side Effects")
st.write("Type the name of a medicine to learn about its potential side effects.")

user_input = st.text_input("Enter medicine name:")
if st.button("Check Side Effects"):
    if user_input.strip():
        with st.spinner("Checking with AI..."):
            try:
                reply = ask_hf(
                    prompt=f"What are the common side effects of {user_input.strip()}?",
                    system_prompt=(
                        "You are a helpful medical assistant. Provide brief, "
                        "clear information about medicine side effects. Always "
                        "remind the user to consult a doctor."
                    ),
                    model=selected_model,
                    max_tokens=300,
                )
                st.info(reply)
            except RuntimeError as e:
                st.error(str(e))
    else:
        st.warning("Please enter a medicine name.")

st.caption(
    "This app provides general information only and is not a substitute for "
    "professional medical advice. Always consult a doctor or pharmacist."
)