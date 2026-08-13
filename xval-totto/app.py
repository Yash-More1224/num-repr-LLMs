import json
import os
import streamlit as st

st.set_page_config(
    page_title="ToTTo Model Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, minimalist design
st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .ref-box {
        background-color: #F8FAFC;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        font-size: 1.15rem;
        font-weight: 500;
        color: #0F172A;
        margin-bottom: 1.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .model-card {
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #E2E8F0;
        background-color: #FFFFFF;
    }
    .model-bpe { border-left: 5px solid #10B981; }
    .model-digit { border-left: 5px solid #3B82F6; }
    .model-fone { border-left: 5px solid #F59E0B; }
    .model-xval { border-left: 5px solid #EF4444; }
    
    .model-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 0.4rem;
    }
    .text-bpe { color: #059669; }
    .text-digit { color: #2563EB; }
    .text-fone { color: #D97706; }
    .text-xval { color: #DC2626; }
    
    .gen-text {
        font-size: 1.1rem;
        color: #1E293B;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

@st.cache_data
def load_all_results():
    files = {
        "bpe": os.path.join(RESULTS_DIR, "bpe_totto-200.json"),
        "digit": os.path.join(RESULTS_DIR, "digit_totto-200.json"),
        "fone": os.path.join(RESULTS_DIR, "fone_totto-200.json"),
        "xval": os.path.join(RESULTS_DIR, "xval_totto-200.json"),
    }
    
    data = {}
    for key, path in files.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
                data[key] = {ex["example_id"]: ex for ex in content.get("examples", [])}
        else:
            data[key] = {}
            
    # Base ordering on BPE examples list
    bpe_path = files["bpe"]
    ordered_ids = []
    if os.path.exists(bpe_path):
        with open(bpe_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            ordered_ids = [ex["example_id"] for ex in content.get("examples", [])]

    return data, ordered_ids

data, ordered_ids = load_all_results()

# Session state initialization for index
if "example_idx" not in st.session_state:
    st.session_state.example_idx = 0

total_examples = len(ordered_ids)

# Sidebar navigation
with st.sidebar:
    st.title("⚙️ Navigation")
    
    # Navigation Buttons
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("⬅️ Previous", use_container_width=True):
            if st.session_state.example_idx > 0:
                st.session_state.example_idx -= 1
                st.rerun()
    with col_next:
        if st.button("Next ➡️", use_container_width=True):
            if st.session_state.example_idx < total_examples - 1:
                st.session_state.example_idx += 1
                st.rerun()
                
    st.session_state.example_idx = st.number_input(
        "Example Index:",
        min_value=0,
        max_value=max(0, total_examples - 1),
        value=st.session_state.example_idx,
        step=1
    )
    
    if total_examples > 0:
        st.slider(
            "Progress",
            min_value=0,
            max_value=total_examples - 1,
            value=st.session_state.example_idx,
            key="slider_idx",
            on_change=lambda: st.session_state.update({"example_idx": st.session_state.slider_idx})
        )

# Main Content
st.markdown('<div class="main-header">ToTTo Model Output Comparison</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Example <b>{st.session_state.example_idx + 1}</b> of <b>{total_examples}</b></div>', unsafe_allow_html=True)

if total_examples > 0:
    current_id = ordered_ids[st.session_state.example_idx]
    
    # Get reference text from any model that has this example
    sample_ex = next((data[k][current_id] for k in ["bpe", "digit", "fone", "xval"] if current_id in data[k]), None)
    
    if sample_ex:
        ref_text = sample_ex.get("reference", sample_ex.get("bleu_ref", ""))
        
        st.markdown(f"""
        <div class="ref-box">
            <b>🎯 Target Reference:</b><br/>
            {ref_text}
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Generated Outputs")
        
        models_info = [
            ("bpe", "1. BPE Model", "model-bpe", "text-bpe"),
            ("digit", "2. Digit Model", "model-digit", "text-digit"),
            ("fone", "3. FoNE Model", "model-fone", "text-fone"),
            ("xval", "4. xVal Model", "model-xval", "text-xval"),
        ]
        
        for m_key, m_name, card_class, text_class in models_info:
            ex = data[m_key].get(current_id)
            if ex:
                gen_text = ex.get("generated", ex.get("bleu_hyp", ""))
                st.markdown(f"""
                <div class="model-card {card_class}">
                    <div class="model-title {text_class}">{m_name}</div>
                    <div class="gen-text">{gen_text}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="model-card {card_class}">
                    <div class="model-title {text_class}">{m_name}</div>
                    <div class="gen-text" style="color: #94A3B8; font-style: italic;">Skipped / Exceeded max length for this example</div>
                </div>
                """, unsafe_allow_html=True)
else:
    st.error("No example results found in results directory!")
