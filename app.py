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
    background: #f8fafc; 
    color: #0f172a; 
}

h1 { 
    font-size: 2.4rem !important; 
    font-weight: 800;
}


.stButton > button {
    background: linear-gradient(90deg, #2563eb, #6366f1);
    color: white !important;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    border: none;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    color: #0f172a !important;
}

.metric-card * {
    color: #0f172a !important;
}

div[data-testid="stAlert"],
div[data-testid="stAlert"] p {
    color: #111111 !important;
    font-weight: 500;
}
textarea {
    background-color: #ffffff !important;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    border-radius: 10px;
}
input {
    color: #0f172a !important;
}
div[data-baseweb="base-input"],
div[data-baseweb="textarea"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
}
div[data-baseweb="textarea"] textarea {
    color: #0f172a !important;
    background-color: #ffffff !important;
}
div[data-testid="stFileUploader"] {
    background-color: #ffffff !important;
    border-radius: 10px;
    padding: 10px;
}

div[data-testid="stFileUploader"] * {
    color: #0f172a !important;
}
textarea::placeholder {
    color: #94a3b8 !important;
}

@media (max-width: 768px) {

    div[data-testid="column"] {
        flex: 1 1 100% !important;
        max-width: 100% !important;
    }

    .metric-card {
        margin-bottom: 10px;
    }

    h1 {
        font-size: 1.7rem !important;
    }

    textarea, input {
        font-size: 16px !important;
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.title("💼 AI İş Analiz Sistemi")
st.caption("Semantic CV Matching + Skill Gap Detection")

ilan = st.text_area("📝 İş ilanı")
cv = st.file_uploader("📄 CV (PDF)", type=["pdf"])

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
    if not job or not cv:
        return 0

    emb1 = model.encode(job, convert_to_tensor=True)
    emb2 = model.encode(cv, convert_to_tensor=True)

    score = float(util.cos_sim(emb1, emb2))
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

if st.button("🚀 ANALİZ ET"):

    if not ilan:
        st.warning("İlan giriniz.")
        st.stop()

    with st.spinner("Analiz ediliyor..."):

        cv_text = pdf_oku(cv)

        job_sk = skill_cikar(ilan)
        cv_sk = skill_cikar(cv_text)

        kw = keyword_score(job_sk, cv_sk)
        sem = semantic_score(ilan, cv_text)
        final = hybrid(sem, kw)

        miss = eksik(job_sk, cv_sk)

        st.markdown("---")
        st.header("📊 Sonuçlar")

        col1, col2, col3 = st.columns(3)

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

        st.success(f"Karar: {karar(final)}")

        if kw < 40:
            st.info("⚠ Skill uyumu düşük")
        if sem < 50:
            st.info("⚠ İlan ile CV arasında içerik uyumu düşük seviyede")

        if miss:
            st.info(f"Geliştirilmesi gereken alan: {', '.join(miss)}")
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
