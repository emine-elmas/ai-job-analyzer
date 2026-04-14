import streamlit as st
import re
import pdfplumber
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

st.set_page_config(
    page_title="AI İş Analiz Sistemi",
    page_icon="💼",
    layout="wide"
)

@st.cache_resource
def model_yukle():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

model = model_yukle()

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1f2937, #374151, #111827);
    color: white;
}

.block-container {
    max-width: 1100px;
    margin: auto;
    padding-top: 2rem;
}

.main-title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 800;
}

.sub-title {
    text-align: center;
    color: #67e8f9;
    margin-bottom: 2rem;
}

/* TEXTAREA FIX */
textarea,
textarea:focus,
textarea:active {
    background: transparent !important;
    color: white !important;
    -webkit-text-fill-color: white !important;
}

div[data-baseweb="textarea"] {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    backdrop-filter: blur(10px);
}

div[data-baseweb="textarea"] > div {
    background: transparent !important;
}

div[data-baseweb="textarea"] textarea {
    background: transparent !important;
    color: white !important;
}

textarea::placeholder {
    color: rgba(255,255,255,0.6) !important;
}

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(12px);
}

div[data-testid="stFileUploader"] section {
    background: transparent !important;
}

div[data-testid="stFileUploader"] * {
    color: #67e8f9 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #06b6d4, #8b5cf6);
    color: white !important;
    border-radius: 20px;
    padding: 0.9rem;
    font-weight: 600;
    border: none;
    width: 100%;
}
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    color: white !important;

    height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.metric-card:hover {
    transform: translateY(-4px);
    transition: 0.2s;
}


label {
    color: #cbd5f5 !important;
}

@media (max-width: 768px) {

    .block-container {
        max-width: 100% !important;
        padding-left: 10px;
        padding-right: 10px;
    }

    .main-title {
        font-size: 1.5rem;
    }

    .sub-title {
        font-size: 0.9rem;
    }

    div[data-baseweb="textarea"] {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }

    textarea {
        font-size: 14px !important;
    }

    div[data-testid="stFileUploader"] {
        padding: 15px;
        border-radius: 16px;
    }

    div[data-testid="column"] {
        flex-direction: column !important;
        gap: 10px;
    }

    .metric-card {
        width: 100% !important;
        height: auto !important;
        padding: 18px;
    }

    div[data-testid="stProgress"] > div {
        height: 8px;
    }
            
    div[style*="#16a34a"] {
        font-size: 14px !important;
        padding: 12px !important;
    }

    div[style*="#2563eb"] {
        font-size: 14px !important;
        padding: 12px !important;
    }

    canvas {
        max-width: 100% !important;
    }
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💼 AI İş Analiz Sistemi</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CV’n İşe Ne Kadar Uygun? Eksiklerini Keşfet</div>', unsafe_allow_html=True)

ilan = st.text_area("📝 İş ilanı")
cv = st.file_uploader("📄 CV (PDF)", type=["pdf"])
if cv:
    st.session_state["cv_file"] = cv

YETENEK_MAP = {
    "Frontend": ["html", "css", "javascript", "react"],
    "Backend": ["node", "django", "flask", "api"],
    "Python": ["python", "pandas", "numpy"],
    "SQL": ["sql", "mysql", "postgresql"],
    "AI": ["machine learning", "deep learning", "tensorflow"],
    "Cloud": ["aws", "docker"],
}

YETENEK_MAP.update({
    "Frontend": ["html", "css", "javascript", "react"],
    "Backend": ["node", "django", "flask", "api", "express", "rest"],
    "Python": ["python", "pandas", "numpy", "oop"],
    "SQL": ["sql", "mysql", "postgresql", "database"],
    "AI": ["machine learning", "deep learning", "tensorflow", "scikit-learn", "keras"],
    "Cloud": ["aws", "docker", "kubernetes"],
    "Data": ["data analysis", "data science", "data visualization"],
    "DevOps": ["ci/cd", "github actions", "linux", "docker"],
    "Mobile": ["flutter", "dart", "android", "ios"],
    "Tools": ["git", "github", "postman"]
})

ROADMAP_MAP = {
    "html": [
        "Semantic HTML yapısını öğren (header, section, article)",
        "Form yapıları ve input validation çalış",
        "Responsive layout için proper structure oluştur"
    ],
    "css": [
        "Flexbox ve Grid sistemini öğren",
        "Responsive design (media query) çalış",
        "Modern UI için bir landing page tasarla"
    ],
    "javascript": [
        "ES6+ özelliklerini öğren (arrow function, map, filter)",
        "DOM manipülasyonu pratikleri yap",
        "API'den veri çekip ekrana basan mini proje geliştir"
    ],
    "react": [
        "Component yapısını ve props/state mantığını öğren",
        "useEffect & useState hook'larını kavra",
        "API bağlantılı küçük bir dashboard geliştir"
    ],
    "node": [
        "Node.js event loop ve async mantığını öğren",
        "Express.js ile REST API geliştir",
        "JWT authentication ekleyerek login sistemi kur"
    ],
    "django": [
        "MTV mimarisini öğren",
        "Django ORM ile CRUD işlemleri yap",
        "Auth sistemi olan mini bir backend projesi geliştir"
    ],
    "flask": [
        "Basit REST API oluştur",
        "SQLAlchemy ile veritabanı bağla",
        "Deploy edilmiş bir Flask API yayına al"
    ],
    "api": [
        "RESTful API prensiplerini öğren",
        "HTTP status code'ları detaylı öğren",
        "Swagger/OpenAPI ile dokümantasyon hazırla"
    ],
    "python": [
        "OOP (class, inheritance) konularını öğren",
        "Virtual environment kullanmayı öğren",
        "Dosya işlemleri ve exception handling pratiği yap"
    ],
    "pandas": [
        "DataFrame manipülasyonu öğren",
        "Groupby & aggregation pratikleri yap",
        "Gerçek bir CSV dataset üzerinde analiz yap"
    ],
    "numpy": [
        "Array yapısını ve broadcasting mantığını öğren",
        "Matrix işlemleri pratiği yap",
        "Numpy + Pandas ile veri temizleme projesi geliştir"
    ],
    "sql": [
        "SELECT, JOIN, GROUP BY konularını öğren",
        "Subquery ve index mantığını kavra",
        "LeetCode SQL soruları çöz"
    ],
    "mysql": [
        "MySQL kurulumu ve schema tasarımı öğren",
        "Foreign key ilişkileri kur",
        "Gerçek bir backend projesine MySQL bağla"
    ],
    "postgresql": [
        "PostgreSQL veri tiplerini öğren",
        "Index ve performans optimizasyonu çalış",
        "Backend uygulamasıyla entegre et"
    ],
    "machine learning": [
        "Linear Regression & Classification algoritmalarını öğren",
        "Scikit-learn ile model geliştir",
        "Kaggle dataset üzerinde proje yap",
        "Modeli Streamlit ile deploy et"
    ],
    "deep learning": [
        "Neural network temel mantığını öğren",
        "TensorFlow veya PyTorch ile basit model kur",
        "Image veya NLP mini proje geliştir"
    ],
    "tensorflow": [
        "Tensor mantığını ve computational graph yapısını öğren",
        "Basit bir classification modeli geliştir",
        "Model kaydetme & yükleme işlemlerini öğren"
    ],
    "aws": [
        "EC2 ve S3 servislerini öğren",
        "Basit bir uygulamayı AWS EC2'ye deploy et",
        "IAM ve security group mantığını kavra"
    ],
    "docker": [
        "Dockerfile yazmayı öğren",
        "Bir backend projesini containerize et",
        "Docker Compose ile multi-container yapı kur"
    ]
}
def temizle(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9çğıöşü\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def pdf_oku(file):
    text = ""
    if file:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    return text

def skill_cikar(text):
    text = temizle(text)
    skills = set()

    for keywords in YETENEK_MAP.values():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text):
                skills.add(kw)

    return skills

def keyword_score(job_sk, cv_sk):
    if not job_sk:
        return 0
    return round((len(job_sk & cv_sk) / len(job_sk)) * 100)

def semantic_score(job, cv):

        job = job[:4000]
        cv = cv[:4000]

        if not job or not cv:
           return 0

        emb1 = model.encode(job, convert_to_tensor=True, normalize_embeddings=True)
        emb2 = model.encode(cv, convert_to_tensor=True, normalize_embeddings=True)

        score = util.cos_sim(emb1, emb2).item()
        return round(score * 100)

def hybrid(sem, key):
    return round(sem * 0.7 + key * 0.3)

def eksik(job_sk, cv_sk):
    return list(job_sk - cv_sk)

def karar(score):
    if score >= 80:
        return "🔥 Mükemmel Uyum"
    elif score >= 50:
        return "⚠ Orta Seviye Uyum"
    else:
        return "❌ Düşük Uyum"

analiz = st.button("🚀 ANALİZ ET")

if analiz:

    if not ilan:
        st.warning("İlan giriniz.")
        st.stop()

    if "cv_file" not in st.session_state:
        st.warning("Lütfen CV yükleyin.")
        st.stop()

    cv = st.session_state["cv_file"]

    with st.spinner("Analiz ediliyor..."):

        cv_text = pdf_oku(cv)

        if not cv_text:
            st.error("❌ CV okunamadı!")
            st.stop()

        job_sk = skill_cikar(ilan)
        cv_sk = skill_cikar(cv_text)

        kw = keyword_score(job_sk, cv_sk)
        sem = semantic_score(ilan, cv_text)
        final = hybrid(sem, kw)

        miss = eksik(job_sk, cv_sk)

        st.markdown("---")
        st.header("📊 Sonuçlar")

        col1, col2, col3 = st.columns(3, gap="small")

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Genel Uyum Skoru</h3>
                <h2>%{final}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>İçerik Benzerliği</h3>
                <h2>%{sem}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Teknik Beceri Eşleşmesi</h3>
                <h2>%{kw}</h2>
            </div>
            """, unsafe_allow_html=True)

        st.progress(final / 100)

        st.markdown(f"""
        <div style="
        background: linear-gradient(90deg, #16a34a, #22c55e);
        padding: 14px;
        border-radius: 12px;
        font-weight: 600;
        color: white;
        text-align:center;
        ">
        Karar: {karar(final)}
        </div>
        """, unsafe_allow_html=True)

        if kw < 40:
            st.info("⚠ Skill uyumu düşük")
        if sem < 45:
            st.info("⚠ İlan ile CV arasında içerik uyumu düşük seviyede")

        if miss:
            st.markdown(f"""
            <div style="
            background: linear-gradient(90deg, #2563eb, #3b82f6);
            padding: 16px;
            border-radius: 12px;
            font-weight: 600;
            color: white;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 10px;
            ">
            Geliştirilmesi gereken alan: {', '.join(miss)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("Eksik yok 🎉")

        st.subheader("📊 Skill Analizi")

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["Uyumlu Yetenekler", "Eksik Yetenekler"], [len(job_sk & cv_sk), len(miss)])

        ax.set_title("Skill Uyumu", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)

        st.pyplot(fig, use_container_width=False)

        if miss:
            st.subheader("🧭 Gelişim Planı")

        for m in miss[:3]:
            st.subheader(f"👉 {m}")
            key = m.lower().strip()

            if key in ROADMAP_MAP:
                for step in ROADMAP_MAP[key]:
                    st.write(f"• {step}")
            else:
                st.write("• Temel konuları öğren")
                st.write("• Mini proje yap")
                st.write("• Pratik yaparak geliştir")
