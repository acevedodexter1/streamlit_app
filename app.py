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
# Both of these have been confirmed working via the Novita provider.
MODEL_OPTIONS = {
    "DeepSeek-V4.1-Flash (recommended, via Novita)": "deepseek-ai/DeepSeek-V4.1-Flash",
    "GLM-5.3-Flash (via Novita)": "zai-org/GLM-5.3-Flash",
}

DATA_PATH = "medicine_schedule.csv"

st.set_page_config(page_title="Medicine System", layout="wide", page_icon="💊")

# =========================================================
# Visual design
# =========================================================
# A calm, clinical look: deep navy background, a single muted teal accent
# used sparingly (buttons, headings), and one legible font throughout.
# The base colors also live in .streamlit/config.toml as the native Streamlit
# theme, so the look holds even if a future Streamlit update changes how
# these CSS selectors render.
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label,
.stApp .stMarkdown, .stButton button, .stTextInput input {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.app-header h1 {
    margin-bottom: 0.1rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.app-subtitle {
    color: #93A5B1;
    font-size: 0.95rem;
    margin-top: 0;
    margin-bottom: 1.5rem;
}

.stButton > button[kind="primary"] {
    background-color: #4FA8A3;
    border: none;
    color: #0B141B;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover {
    background-color: #62BDB7;
    color: #0B141B;
}

section[data-testid="stSidebar"] h2 {
    font-size: 1.05rem;
    margin-top: 1.2rem;
    margin-bottom: 0.4rem;
}

hr {
    border-color: rgba(255,255,255,0.08);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header"><h1>💊 Medicine Reminder & Interaction Checker</h1></div>
    <p class="app-subtitle">Track your doses, catch scheduling conflicts, and ask the AI about interactions or side effects.</p>
    """,
    unsafe_allow_html=True,
)


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
    submitted = st.form_submit_button("Add to Schedule", type="primary")

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

# =========================================================
# Sidebar: Remove Medicine (new)
# =========================================================
st.sidebar.divider()
st.sidebar.header("🗑️ Remove Medicine")
if st.session_state.df.empty:
    st.sidebar.caption("No medicines yet — add one above first.")
else:
    remove_options = st.session_state.df.index.tolist()
    med_to_remove = st.sidebar.selectbox(
        "Choose a medicine to remove",
        remove_options,
        format_func=lambda i: (
            f"{st.session_state.df.loc[i, 'Medicine']} "
            f"({st.session_state.df.loc[i, 'Dosage']}, {st.session_state.df.loc[i, 'Time']})"
        ),
        key="remove_select",
    )
    if st.sidebar.button("Remove Selected Medicine"):
        removed_name = st.session_state.df.loc[med_to_remove, "Medicine"]
        st.session_state.df = st.session_state.df.drop(index=med_to_remove).reset_index(drop=True)
        save_df()
        st.sidebar.success(f"Removed {removed_name} from your schedule.")
        st.rerun()

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
    st.info("No medicines yet — add your first one from the sidebar.")
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
        with st.container(border=True):
            st.markdown(f"⏰ **Next up ({when}):** {row['Medicine']} ({row['Dosage']}) at {row['Time']}")

    # --- Same-time conflict check (no AI needed, instant) ---
    dupe_times = display_df["Time"][display_df["Time"].duplicated(keep=False)]
    if not dupe_times.empty:
        conflict_groups = display_df[display_df["Time"].isin(dupe_times.unique())]
        for t, group in conflict_groups.groupby("Time"):
            meds = ", ".join(group["Medicine"])
            st.warning(f"🕒 Scheduled at the same time ({t}): {meds}")

    st.caption("Edit cells directly, or use 'Remove Medicine' in the sidebar to delete a dose.")
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
with st.container(border=True):
    st.write(
        "Ask the AI to review every medicine currently in your schedule for "
        "well-known interactions or timing considerations."
    )
    if st.button("Check My Full Schedule for Interactions", type="primary"):
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
with st.container(border=True):
    st.write("Type the name of a medicine to learn about its potential side effects.")

    user_input = st.text_input("Enter medicine name:")
    if st.button("Check Side Effects", type="primary"):
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