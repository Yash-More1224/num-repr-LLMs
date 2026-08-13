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
        margin-bottom: 1rem;
    }
    .id-badge {
        display: inline-block;
        background-color: #E2E8F0;
        color: #334155;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .ref-box {
        background-color: #F8FAFC;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        font-size: 1.15rem;
        font-weight: 500;
        color: #0F172A;
        margin-bottom: 1.5rem;
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
        "fone": os.path.join(RESULTS_DIR, "fone_totto-fixed-200.json"),
        "xval": os.path.join(RESULTS_DIR, "xval_totto-200-fixed-gen.json"),
    }
    
    data = {}
    for key, path in files.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
                data[key] = {ex["example_id"]: ex for ex in content.get("examples", [])}
        else:
            data[key] = {}
            
    # Partition into examples with numbers vs without numbers based on BPE list
    bpe_path = files["bpe"]
    ids_with_numbers = []
    ids_without_numbers = []
    
    if os.path.exists(bpe_path):
        with open(bpe_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            for ex in content.get("examples", []):
                eid = ex["example_id"]
                ref_nums = ex.get("ref_nums", [])
                if len(ref_nums) > 0:
                    ids_with_numbers.append(eid)
                else:
                    ids_without_numbers.append(eid)

    return data, ids_with_numbers, ids_without_numbers

data, ids_with_numbers, ids_without_numbers = load_all_results()

# Header
st.markdown('<div class="main-header">ToTTo Model Output Comparison</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs([
    f"🔢 Examples with Numbers ({len(ids_with_numbers)})",
    f"📝 Examples without Numbers ({len(ids_without_numbers)})"
])

def render_example_viewer(target_ids, session_key_prefix):
    total = len(target_ids)
    if total == 0:
        st.info("No examples in this category.")
        return
        
    idx_key = f"{session_key_prefix}_idx"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0

    # Navigation controls
    col_prev, col_idx, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("⬅️ Previous", key=f"{session_key_prefix}_prev", use_container_width=True):
            if st.session_state[idx_key] > 0:
                st.session_state[idx_key] -= 1

    with col_next:
        if st.button("Next ➡️", key=f"{session_key_prefix}_next", use_container_width=True):
            if st.session_state[idx_key] < total - 1:
                st.session_state[idx_key] += 1

    with col_idx:
        st.number_input(
            "Example Index",
            min_value=0,
            max_value=total - 1,
            step=1,
            key=idx_key
        )

    # Draggable Slider
    st.slider(
        "Drag to jump to example:",
        min_value=0,
        max_value=total - 1,
        key=f"{session_key_prefix}_slider",
        value=st.session_state[idx_key],
        on_change=lambda key=idx_key, s_key=f"{session_key_prefix}_slider": st.session_state.update({key: st.session_state[s_key]})
    )

    current_index = st.session_state[idx_key]
    current_id = target_ids[current_index]

    st.markdown(f'<div class="sub-header">Showing Example <b>{current_index + 1}</b> of <b>{total}</b> &nbsp;|&nbsp; Example ID: <span class="id-badge">{current_id}</span></div>', unsafe_allow_html=True)

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

with tab1:
    render_example_viewer(ids_with_numbers, "num")

with tab2:
    render_example_viewer(ids_without_numbers, "no_num")
