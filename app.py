import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# ── NLTK setup ────────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ── Model (cached) ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ── Role keywords & skills ─────────────────────────────────────────────────────
ROLE_KEYWORDS = {
    "Accountant":          ["accounting", "ledger", "tax", "audit", "financial"],
    "Advocate":            ["law", "legal", "litigation", "court", "advocate"],
    "Arts":                ["artist", "design", "creative", "painting", "art"],
    "Banking":             ["bank", "loan", "credit", "risk", "finance"],
    "Business Development":["business development", "sales strategy", "client acquisition"],
    "Chef":                ["chef", "cooking", "kitchen", "menu", "culinary"],
    "Construction":        ["construction", "site", "civil", "project", "building"],
    "Consultant":          ["consulting", "advisory", "strategy", "client"],
    "Digital Media":       ["digital marketing", "seo", "social media", "content"],
    "Engineering":         ["engineering", "mechanical", "electrical", "civil"],
    "Finance":             ["finance", "investment", "portfolio", "analysis"],
    "IT":                  ["python", "java", "software", "machine learning", "data"],
    "Sales":               ["sales", "target", "revenue", "customer", "negotiation"],
}

ROLE_SKILLS = {
    "Accountant":          ["accounting","financial reporting","audit","tax","ledger","accounts payable","accounts receivable","balance sheet","budgeting","forecasting","erp","tally","quickbooks","excel"],
    "Advocate":            ["legal","litigation","court","contract drafting","compliance","legal research","case law","corporate law","civil law","criminal law","arbitration","negotiation"],
    "Arts":                ["design","creative","illustration","painting","sketching","photoshop","adobe illustrator","visual design","animation","graphic design","storyboarding"],
    "Banking":             ["banking","loan processing","credit analysis","risk assessment","financial services","investment","mortgage","retail banking","corporate banking","compliance","kyc"],
    "Business Development":["business development","lead generation","client acquisition","sales strategy","market research","negotiation","relationship management","crm","revenue growth"],
    "Chef":                ["cooking","culinary","menu planning","food preparation","kitchen management","food safety","baking","recipe development","inventory management"],
    "Construction":        ["construction","site management","civil engineering","project management","autocad","blueprints","safety compliance","building materials","structural design"],
    "Consultant":          ["consulting","strategy","business analysis","problem solving","stakeholder management","process improvement","data analysis","presentation","client management"],
    "Digital Media":       ["digital marketing","seo","social media","content creation","google analytics","ads","branding","email marketing","campaign management","copywriting"],
    "Engineering":         ["engineering","mechanical","electrical","civil","autocad","solidworks","matlab","design","testing","manufacturing","project engineering"],
    "Finance":             ["finance","financial analysis","investment","portfolio management","risk management","valuation","budgeting","forecasting","excel","financial modeling"],
    "IT":                  ["python","java","c++","sql","machine learning","data analysis","deep learning","nlp","tensorflow","pandas","numpy","api","cloud","aws","docker","git"],
    "Sales":               ["sales","lead generation","customer relationship","negotiation","closing deals","target achievement","crm","communication","marketing","revenue"],
}

# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^\x00-\x7f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# ── PDF extraction ─────────────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    return " ".join(page.extract_text() or "" for page in reader.pages)

# ── Role detection ─────────────────────────────────────────────────────────────
def detect_role(text: str) -> str:
    text = text.lower()
    scores = {role: sum(1 for kw in kws if kw in text) for role, kws in ROLE_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Unknown"

# ── Skill extraction ───────────────────────────────────────────────────────────
def extract_skills(text: str, role: str) -> set:
    text = text.lower()
    return {skill for skill in ROLE_SKILLS.get(role, []) if skill in text}

# ── Skill match score ──────────────────────────────────────────────────────────
def skill_match_score(jd: str, resume: str) -> float:
    jd_words = set(jd.split())
    return len(jd_words & set(resume.split())) / len(jd_words) if jd_words else 0.0

# ── Core scoring ───────────────────────────────────────────────────────────────
def score_resumes(job_description: str, resumes: list[str], role: str):
    """Returns list of dicts with rank, bert_score, skill_score, final_score, matched, missing."""
    jd_clean   = clean_text(job_description)
    res_clean  = [clean_text(r) for r in resumes]
    jd_skills  = extract_skills(jd_clean, role)

    jd_emb  = model.encode(jd_clean).reshape(1, -1)
    res_emb = np.array(model.encode(res_clean))
    if res_emb.ndim == 1:
        res_emb = res_emb.reshape(1, -1)

    bert_scores = cosine_similarity(jd_emb, res_emb)[0]

    results = []
    for i, res in enumerate(res_clean):
        res_skills  = extract_skills(res, role)
        matched     = jd_skills & res_skills
        missing     = jd_skills - res_skills
        sk_score    = len(matched) / len(jd_skills) if jd_skills else 0.0
        final       = 0.7 * bert_scores[i] + 0.3 * sk_score
        results.append({
            "index":       i + 1,
            "bert_score":  float(bert_scores[i]),
            "skill_score": sk_score,
            "final_score": final,
            "matched":     sorted(matched),
            "missing":     sorted(missing),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    for rank, r in enumerate(results, 1):
        r["rank"] = rank
    return results

# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Resume Similarity Screener", page_icon="📄", layout="wide")
st.title("📄 Resume Similarity Screener")
st.caption("Upload PDF resumes, provide a job description, and get ranked candidates instantly.")
with st.expander('INFO ℹ️'):
    st.subheader("Developed by :red[Aswin S]")
    col1,col2 = st.columns(2)
    with col1:
        st.link_button(":blue[LinkedIn]","https://www.linkedin.com/in/aswin-sgl")
    with col2:
        st.link_button(":red[GitHub]","https://github.com/aswins11401-sudo")
# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    target_role = st.selectbox("Target Role", sorted(ROLE_KEYWORDS.keys()))
    jd_source   = st.radio("Job Description Source", ["Paste manually", "Upload your posting csv"])
# ── Job Description ────────────────────────────────────────────────────────────
st.subheader("📋 Job Description")


job_description = ""
if jd_source == "Paste manually":
    job_description = st.text_area("Paste job description here", height=200,
                                   placeholder="e.g. We are looking for a Python developer with ML experience...")
else:
    csv_file = st.file_uploader("Upload postings.csv", type=["csv"])
    if csv_file:
        df_jd = pd.read_csv(csv_file, engine="python")
        filtered = df_jd[df_jd["title"].str.contains(target_role, case=False, na=False)]
        if not filtered.empty:
            st.success(f"Found {len(filtered)} matching job descriptions for '{target_role}'.")
            job_description = filtered["description"].iloc[0]
            with st.expander("Preview JD"):
                st.write(job_description[:1000] + "...")
        else:
            st.warning(f"No job descriptions found for '{target_role}' in the CSV.")

# ── Resume Upload — PDF only ───────────────────────────────────────────────────
st.subheader("📁 Upload Resumes (PDF)")
st.caption("You can select and upload multiple PDF files at once.")

uploaded_pdfs = st.file_uploader(
    "Drop PDF resumes here or click to browse",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

raw_resumes  = []
resume_names = []

if uploaded_pdfs:
    st.markdown(f"**{len(uploaded_pdfs)} file(s) uploaded:**")
    preview_cols = st.columns(min(len(uploaded_pdfs), 4))
    for idx, pdf in enumerate(uploaded_pdfs):
        with preview_cols[idx % 4]:
            st.markdown(
                f"""
                <div style="border:1px solid #e0e0e0; border-radius:8px; padding:10px; text-align:center; background:#f9f9f9;">
                    📄<br><small><b>{pdf.name}</b></small><br>
                    <small style="color:grey;">{pdf.size // 1024} KB</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        text = extract_pdf_text(pdf)
        raw_resumes.append(text)
        resume_names.append(pdf.name)

# ── Run ────────────────────────────────────────────────────────────────────────
st.divider()
run = st.button("🚀 Screen Resumes", type="primary",
                disabled=(not job_description or not raw_resumes))

if run:
    valid_resumes = [(name, text) for name, text in zip(resume_names, raw_resumes) if text.strip()]
    if not valid_resumes:
        st.error("Could not extract text from the uploaded PDFs. Please ensure the files are not scanned images.")
    else:
        names, texts = zip(*valid_resumes)
        mode = "Single Resume Mode" if len(texts) == 1 else f"Multiple Resume Mode ({len(texts)} resumes)"
        st.info(f"🔍 {mode} — Role: **{target_role}**")

        with st.spinner("Encoding and scoring..."):
            results = score_resumes(job_description, list(texts), target_role)

        # ── Summary table ──────────────────────────────────────────────────────
        st.subheader("🏆 Ranked Results")
        table = []
        for r in results:
            name = names[r["index"] - 1]
            table.append({
                "Rank":           r["rank"],
                "Resume":         name,
                "Final Score":    f"{r['final_score']*100:.1f}%",
                "BERT Score":     f"{r['bert_score']*100:.1f}%",
                "Skill Score":    f"{r['skill_score']*100:.1f}%",
                "Matched Skills": ", ".join(r["matched"]) or "—",
                "Missing Skills": ", ".join(r["missing"]) or "—",
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        # ── Per-resume detail ──────────────────────────────────────────────────
        st.subheader("📊 Detailed Breakdown")
        for r in results:
            name = names[r["index"] - 1]
            with st.expander(f"#{r['rank']} — {name}  |  {r['final_score']*100:.1f}%"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Final Score",  f"{r['final_score']*100:.1f}%")
                c2.metric("BERT Score",   f"{r['bert_score']*100:.1f}%")
                c3.metric("Skill Match",  f"{r['skill_score']*100:.1f}%")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Matched Skills**")
                    st.write(r["matched"] if r["matched"] else ["None detected"])
                with col2:
                    st.markdown("**❌ Missing Skills**")
                    st.write(r["missing"] if r["missing"] else ["None"])
                    
with st.expander('INFO'):
    st.markdown('''
                BERT score is calculated based on the :red[Cosine similarity] between JD and Resumes \n
                Final Scores are calculated as :blue[0.7 * BERT score] + :red[0.3 * Skill score] \n
                Skill score is calculated Based on the :blue[number of skills matched] / :red[number of skills in JD] \n
                :blue[Matched Skills] and :red[Missing Skills] should give a clear understanding of the ranking of the resume''')