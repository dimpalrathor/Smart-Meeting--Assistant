import streamlit as st
import requests
import os
from google import genai

BACKEND_URL = "https://smart-meeting-assistant-edfi.onrender.com/summarize"


st.set_page_config(page_title="Smart Meeting Assistant", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif; }

:root {
    --primary: #a78bfa;
    --glass-bg: rgba(139, 92, 246, 0.05);
    --glass-border: rgba(167, 139, 250, 0.2);
    --text-primary: #ffffff;
    --text-secondary: #e9d5ff;
}

.stApp { background: radial-gradient(ellipse at top, #1a0b2e 0%, #0a0a0f 50%, #000 100%); background-attachment: fixed; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 2rem; max-width: 1400px; }

h1 {
    background: linear-gradient(135deg, #a78bfa 0%, #c4b5fd 50%, #ddd6fe 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 4.5rem !important; font-weight: 900 !important; text-align: center;
    filter: drop-shadow(0 0 60px rgba(167, 139, 250, 0.6));
}

.subtitle { text-align: center; color: var(--text-secondary); font-size: 1.5rem; 
    margin-bottom: 3rem; text-shadow: 0 0 20px rgba(167, 139, 250, 0.3); }

h2 { color: var(--text-primary) !important; font-weight: 800 !important; 
    font-size: 2.5rem !important; text-shadow: 0 0 30px rgba(167, 139, 250, 0.4); }

.stButton > button {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(167, 139, 250, 0.15));
    color: var(--text-primary); border: 1.5px solid var(--glass-border);
    padding: 1.2rem 2.5rem; font-size: 1.1rem; font-weight: 700; border-radius: 16px;
    backdrop-filter: blur(20px); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    transition: all 0.4s ease; width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(167, 139, 250, 0.3), rgba(196, 181, 253, 0.2));
    border-color: rgba(167, 139, 250, 0.4); transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(167, 139, 250, 0.3);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 1rem; background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(167, 139, 250, 0.05));
    padding: 1rem; border-radius: 20px; backdrop-filter: blur(30px);
    border: 1.5px solid var(--glass-border); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 14px; color: #c4b5fd;
    padding: 1.2rem 2.5rem; font-weight: 700; font-size: 1.1rem;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(167, 139, 250, 0.1); color: var(--text-secondary);
    transform: translateY(-2px);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(167, 139, 250, 0.25), rgba(139, 92, 246, 0.2)) !important;
    color: var(--text-primary) !important; border: 1px solid rgba(167, 139, 250, 0.4) !important;
    box-shadow: 0 6px 24px rgba(167, 139, 250, 0.3);
}

.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.05), rgba(10, 10, 15, 0.8)) !important;
    border: 1.5px solid var(--glass-border) !important; border-radius: 14px !important;
    color: var(--text-primary) !important; backdrop-filter: blur(20px);
    font-size: 1.05rem !important; padding: 1.2rem 1.5rem !important;
}

.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
    border-color: rgba(167, 139, 250, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.15) !important;
}

.stTextInput label, .stTextArea label, .stNumberInput label {
    color: var(--text-primary) !important; font-weight: 700 !important;
    font-size: 1.05rem !important; text-shadow: 0 0 10px rgba(167, 139, 250, 0.3);
}

.stFileUploader {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.05), rgba(167, 139, 250, 0.03));
    border-radius: 24px; padding: 3rem; border: 2px dashed var(--glass-border);
    backdrop-filter: blur(30px);
}

.stFileUploader:hover {
    border-color: rgba(167, 139, 250, 0.4);
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(167, 139, 250, 0.05));
    box-shadow: 0 8px 32px rgba(167, 139, 250, 0.2);
}

.stFileUploader label {
    color: var(--text-primary) !important; font-weight: 700 !important;
    text-shadow: 0 0 20px rgba(167, 139, 250, 0.3);
}

.stAlert {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(10, 10, 15, 0.8)) !important;
    backdrop-filter: blur(30px); border-radius: 16px;
    border: 1.5px solid var(--glass-border); color: var(--text-primary) !important;
    padding: 1.5rem 2rem !important; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.stSuccess {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(10, 10, 15, 0.8)) !important;
    border-left: 4px solid #10b981 !important;
}

.stInfo {
    background: linear-gradient(135deg, rgba(167, 139, 250, 0.15), rgba(10, 10, 15, 0.8)) !important;
    border-left: 4px solid var(--primary) !important;
}

audio {
    width: 100%; border-radius: 14px;
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(10, 10, 15, 0.9));
    border: 1.5px solid var(--glass-border); backdrop-filter: blur(20px);
}

.info-box {
    background: linear-gradient(135deg, rgba(167, 139, 250, 0.12), rgba(139, 92, 246, 0.08));
    border-left: 4px solid var(--primary); padding: 2rem 2.5rem; border-radius: 16px;
    backdrop-filter: blur(30px); border: 1.5px solid rgba(167, 139, 250, 0.2);
    box-shadow: 0 8px 32px rgba(167, 139, 250, 0.15); margin: 2rem 0;
}

.task-card {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(10, 10, 15, 0.8));
    padding: 2rem 2.5rem; border-radius: 18px; margin-bottom: 1.5rem;
    border: 1.5px solid var(--glass-border); backdrop-filter: blur(30px);
    transition: all 0.4s ease; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.task-card:hover {
    transform: translateX(10px); box-shadow: 0 12px 48px rgba(167, 139, 250, 0.25);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10, 10, 15, 0.98), rgba(5, 5, 8, 0.98)) !important;
    backdrop-filter: blur(40px); border-right: 1.5px solid var(--glass-border);
    box-shadow: 4px 0 32px rgba(167, 139, 250, 0.2);
}

[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    text-shadow: 0 0 20px rgba(167, 139, 250, 0.4);
    margin-bottom: 1.5rem !important;
}

[data-testid="stSidebar"] .stTextInput input {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(10, 10, 15, 0.9)) !important;
    border: 1.5px solid var(--glass-border) !important;
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text-secondary) !important;
    font-size: 1rem !important;
    line-height: 1.8 !important;
}

[data-testid="stSidebar"] hr {
    margin: 2rem 0 !important;
    background: linear-gradient(90deg, transparent, rgba(167, 139, 250, 0.3), transparent) !important;
}

/* Sidebar toggle button styling */
[data-testid="collapsedControl"] {
    background: linear-gradient(135deg, rgba(167, 139, 250, 0.3), rgba(139, 92, 246, 0.2)) !important;
    border: 1.5px solid rgba(167, 139, 250, 0.4) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 4px 16px rgba(167, 139, 250, 0.3) !important;
}

.stMarkdown { color: var(--text-secondary) !important; line-height: 2; }
.stMarkdown p { color: var(--text-secondary) !important; font-size: 1.1rem; }

hr {
    border: none; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(167, 139, 250, 0.3), transparent);
    margin: 3.5rem 0;
}
</style>
""", unsafe_allow_html=True)

if "step" not in st.session_state: st.session_state.step = 1
if "meeting_plan" not in st.session_state: st.session_state.meeting_plan = {}
if "audio_data" not in st.session_state: st.session_state.audio_data = None
if "preparation_done" not in st.session_state: st.session_state.preparation_done = False

st.markdown("<h1> Smart Meeting Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Plan → Prepare → Record → Analyze with AI</p>", unsafe_allow_html=True)

# Prominent API Key Entry Area (Main Content)
st.markdown("""
<div style='background: linear-gradient(135deg, rgba(167, 139, 250, 0.15), rgba(139, 92, 246, 0.1));
            padding: 2rem 2.5rem; border-radius: 20px; border: 2px solid rgba(167, 139, 250, 0.3);
            backdrop-filter: blur(30px); margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(167, 139, 250, 0.2);'>
    <h3 style='color: #ffffff; margin-bottom: 1rem; text-shadow: 0 0 20px rgba(167, 139, 250, 0.4);'>
         API Configuration
    </h3>
    <p style='color: #e9d5ff; margin-bottom: 1.5rem;'>
        Enter your API keys to enable AI features
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(167, 139, 250, 0.12), rgba(139, 92, 246, 0.08));
                padding: 1rem; border-radius: 12px; margin-bottom: 1rem;
                border: 1px solid rgba(167, 139, 250, 0.2);'>
        <strong style='color: #a78bfa;'> Required:</strong>
        <span style='color: #e9d5ff; font-size: 0.9rem;'> Get from <a href='https://aistudio.google.com/app/apikey' target='_blank' style='color: #c4b5fd;'>Google AI Studio</a></span>
    </div>
    """, unsafe_allow_html=True)
    gemini_api_key = st.text_input("Gemini API Key", type="password", key="gemini_main", 
                                     placeholder="Enter your Gemini API key...")

with col2:
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(167, 139, 250, 0.08), rgba(139, 92, 246, 0.05));
                padding: 1rem; border-radius: 12px; margin-bottom: 1rem;
                border: 1px solid rgba(167, 139, 250, 0.15);'>
        <strong style='color: #a78bfa;'> Optional:</strong>
        <span style='color: #e9d5ff; font-size: 0.9rem;'> For web search enhancement</span>
    </div>
    """, unsafe_allow_html=True)
    serper_api_key = st.text_input("Serper API Key (Optional)", type="password", key="serper_main",
                                     placeholder="Optional: Serper API key...")

# API Status Indicator
api_status = " API Connected" if gemini_api_key else " API Key Required"
api_color = "#10b981" if gemini_api_key else "#ef4444"

st.markdown(f"""
<div style='background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(10, 10, 15, 0.8));
            padding: 1rem 2rem; border-radius: 14px; border: 1.5px solid var(--glass-border);
            backdrop-filter: blur(30px); text-align: center; margin-bottom: 2rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);'>
    <span style='color: {api_color}; font-weight: 700; font-size: 1.1rem;'>{api_status}</span>
</div>
""", unsafe_allow_html=True)
nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    if st.button(" Plan", key="nav_plan"): st.session_state.step = 1; st.rerun()
with nav2:
    if st.button(" Prepare", key="nav_prepare"): st.session_state.step = 2; st.rerun()
with nav3:
    if st.button(" Record", key="nav_record"): st.session_state.step = 3; st.rerun()
with nav4:
    if st.button(" Analyze", key="nav_analyze"): st.session_state.step = 4; st.rerun()


st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.step == 1:
    st.markdown("<h2> Step 1: Plan Your Meeting</h2>", unsafe_allow_html=True)
    
    with st.form("plan_form"):
        c1, c2 = st.columns(2)
        with c1:
            company = st.text_input(" Company Name")
            objective = st.text_input(" Objective")
            duration = st.number_input(" Duration (min)", 15, 180, 60)
        with c2:
            title = st.text_input(" Title")
            attendees = st.text_area(" Attendees", height=100)
        
        focus = st.text_area(" Focus Areas", height=120)
        
        c1, c2 = st.columns([3, 1])
        with c1: save = st.form_submit_button(" Save Plan", key="form_save_plan")
        with c2: skip = st.form_submit_button(" Skip", key="form_skip_plan")

    
    if save:
        st.session_state.meeting_plan = {"company_name": company, "title": title, "objective": objective, 
                                         "attendees": attendees, "duration": duration, "focus_areas": focus}
        st.session_state.step = 2
        st.success(" Plan saved!")
        st.rerun()
    
    if skip:
        st.session_state.meeting_plan = {"company_name": company or "N/A", "title": title or "Meeting",
                                         "objective": objective or "Discussion", "attendees": attendees or "N/A",
                                         "duration": duration, "focus_areas": focus or "N/A"}
        st.session_state.step = 3
        st.info(" Skipped to recording")
        st.rerun()

elif st.session_state.step == 2:
    st.markdown("<h2> Step 2: AI Preparation</h2>", unsafe_allow_html=True)
    
    if not gemini_api_key:
        st.warning(" Enter Gemini API key in sidebar")
        if st.button("Skip", key="prep_skip"): st.session_state.step = 3; st.rerun()

    else:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
        if serper_api_key: os.environ["SERPER_API_KEY"] = serper_api_key
        
        plan = st.session_state.meeting_plan
        st.markdown(f"""
        <div class='info-box'>
            <strong style='color: #c4b5fd; font-size: 1.2rem;'> Company:</strong> 
            <span style='color: #ffffff; font-size: 1.15rem;'>{plan['company_name']}</span>
            <br><br>
            <strong style='color: #c4b5fd; font-size: 1.2rem;'> Objective:</strong> 
            <span style='color: #ffffff; font-size: 1.15rem;'>{plan['objective']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(" Generate Preparation", key="prep_generate"):
            with st.spinner(" Preparing..."):
                def run_gemini(prompt):
                    try:
                        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
                        resp = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
                        if hasattr(resp, "text") and resp.text: return resp.text.strip()
                        for c in resp.candidates:
                            if c.content.parts and hasattr(c.content.parts[0], "text"):
                                return c.content.parts[0].text.strip()
                        return str(resp)
                    except Exception as e:
                        return f" Error: {e}"
                
                context = run_gemini(f"Analyze meeting context for {plan['company_name']}: {plan['objective']}")
                strategy = run_gemini(f"Create {plan['duration']}-min agenda for {plan['company_name']}: {plan['objective']}")
                brief = run_gemini(f"Executive brief for meeting with {plan['company_name']}: {plan['objective']}")
                
                st.session_state.ai_preparation = {"context": context, "strategy": strategy, "brief": brief}
                st.success(" Preparation complete!")
                st.session_state.preparation_done = True
        
        if "ai_preparation" in st.session_state:
            prep = st.session_state.ai_preparation
            t1, t2, t3 = st.tabs([" Context", " Strategy", " Brief"])
            with t1: st.markdown(prep["context"])
            with t2: st.markdown(prep["strategy"])
            with t3: st.markdown(prep["brief"])
            
            if st.button(" Next", key="prep_next"): st.session_state.step = 3; st.rerun()


elif st.session_state.step == 3:
    st.markdown("<h2> Step 3: Record Meeting</h2>", unsafe_allow_html=True)
    
    t_up, t_rec = st.tabs([" Upload", " Record"])
    
    audio_bytes, filename = None, None
    
    with t_up:
        st.markdown("#### Upload audio file")
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), rgba(10, 10, 15, 0.8));
                    padding: 1.5rem; border-radius: 14px; border: 1.5px solid rgba(167, 139, 250, 0.15);
                    margin: 1.5rem 0; backdrop-filter: blur(20px);'>
            <strong style='color: #a78bfa;'>Supported:</strong>
            <span style='color: #e9d5ff;'> WAV, MP3, M4A, FLAC, OGG, WEBM</span>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Drop file", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"], label_visibility="collapsed")
        
        if uploaded:
            ext = uploaded.name.split('.')[-1].upper()
            size_mb = len(uploaded.getvalue()) / (1024 * 1024)
            st.success(f" Loaded: {uploaded.name}")
            st.markdown(f"""
            <div class='info-box'>
                <strong style='color: #c4b5fd;'> File:</strong> {uploaded.name}<br>
                <strong style='color: #c4b5fd;'> Format:</strong> {ext}<br>
                <strong style='color: #c4b5fd;'> Size:</strong> {size_mb:.2f} MB
            </div>
            """, unsafe_allow_html=True)
            st.audio(uploaded)
            audio_bytes, filename = uploaded.getvalue(), uploaded.name
    
    with t_rec:
        st.markdown("#### Record in real-time")
        st.info(" Allow browser permissions")
        
        fmt = st.selectbox("Format", ["WEBM (Mobile)", "M4A/AAC", "WAV", "MP3", "OGG"], index=0)
        
        st.markdown(f"""
        <div class='info-box'>
            <strong style='color: #22c55e;'> {fmt}:</strong>
            <span style='color: #ffffff;'> {'Best for mobile browsers' if 'WEBM' in fmt else 'iOS/Android native' if 'M4A' in fmt else 'Universal format'}</span>
        </div>
        """, unsafe_allow_html=True)
        
        recorded = st.audio_input(" Record")
        
        if recorded:
            st.success(" Recorded!")
            audio_bytes = recorded.getvalue()
            exts = {"WEBM (Mobile)": "webm", "M4A/AAC": "m4a", "WAV": "wav", "MP3": "mp3", "OGG": "ogg"}
            filename = f"recording.{exts[fmt]}"
            st.audio(audio_bytes)
    
    if audio_bytes:
        st.session_state.audio_data = {"bytes": audio_bytes, "filename": filename}
        if st.button(" Analyze", key="record_analyze"): st.session_state.step = 4; st.rerun()


elif st.session_state.step == 4:
    st.markdown("<h2> Step 4: AI Summary</h2>", unsafe_allow_html=True)
    
    if not st.session_state.audio_data:
        st.error(" No audio found")
        if st.button("← Back", key="summary_back"): st.session_state.step = 3; st.rerun()

    else:
        audio_info = st.session_state.audio_data
        
        if st.button(" Generate Summary", key="summary_generate"):
            with st.spinner(" Analyzing..."):
                try:
                    files = {
                        "audio": (audio_info["filename"], audio_info["bytes"])
                    }
                    resp = requests.post(
                        f"{BACKEND_URL}/summarize",
                        files=files,
                        timeout=600
                    )

                    
                    if resp.status_code == 200:
                        st.session_state.summary_data = resp.json()
                        st.success(" Summary complete!")
                    else:
                        st.error(f" Error: {resp.text}")
                except Exception as e:
                    st.error(f" {e}")
        
        if "summary_data" in st.session_state:
            data = st.session_state.summary_data
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(10, 10, 15, 0.9));
                        padding: 3rem; border-radius: 24px; border: 1.5px solid rgba(167, 139, 250, 0.25);
                        backdrop-filter: blur(30px); box-shadow: 0 12px 48px rgba(167, 139, 250, 0.2);
                        color: #ffffff; line-height: 2;'>
                {data.get("structured_summary", "No summary")}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            t1, t2, t3, t4, t5, t6 = st.tabs([" Transcript", " Actions", " Tasks", " Email", " WhatsApp", " Speakers"])
            
            with t1:
                st.text_area("Transcript", data.get("transcript", ""), height=400, label_visibility="collapsed")
            
            with t2:
                aps = data.get("action_points", [])
                if aps:
                    for i, ap in enumerate(aps, 1):
                        st.markdown(f"""
                        <div class='task-card'>
                            <strong style='color: #a78bfa;'>{i}.</strong> 
                            <span style='color: #ffffff;'>{ap}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No actions")
            
            with t3:
                tasks = data.get("tasks", [])
                if tasks:
                    for i, t in enumerate(tasks, 1):
                        st.markdown(f"""
                        <div class='task-card'>
                            <strong style='color: #a78bfa;'>Task {i}</strong><br>
                            <strong style='color: #c4b5fd;'></strong> {t.get('assignee', 'Unknown')}<br>
                            <strong style='color: #c4b5fd;'></strong> {t.get('task')}<br>
                            <strong style='color: #c4b5fd;'></strong> {t.get('deadline', 'No deadline')}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No tasks")
            
            with t4:
                st.text_area("Email", data.get("followup_email", ""), height=350, label_visibility="collapsed")
            
            with t5:
                st.text_area("WhatsApp", data.get("whatsapp", ""), height=250, label_visibility="collapsed")
            
            with t6:
                speakers = data.get("speakers", [])
                if speakers:
                    for s in speakers: st.markdown(f"- {s}")
                else:
                    st.info("No speakers")
                
                diar = data.get("diarization", [])
                if diar: st.json(diar)
            
            if st.button(" New Meeting", key="new_meeting"):
                for k in list(st.session_state.keys()): del st.session_state[k]
                st.session_state.step = 1
                st.rerun()
