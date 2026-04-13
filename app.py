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
.stApp { background: #f8fafc; color: #0f172a; }
h1 { font-size: 2.5rem !important; font-weight: 800; }
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #6366f1);
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.title("💼 AI İş Analiz Sistemi")
st.caption("Semantic CV Matching + Skill Gap Detection + Akıllı Gelişim Planı")

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


def skill_graph(job_sk, cv_sk):
    match = len(job_sk & cv_sk)
    missing = len(job_sk - cv_sk)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Eşleşen", "Eksik"], [match, missing])
    ax.set_title("Skill Uyumu Analizi" , fontsize=10)
    ax.set_ylabel("Skill Sayısı" , fontsize=8)

    st.pyplot(fig, use_container_width=False)


def roadmap(missing_skills):
    if not missing_skills:
        st.success("🎉 Tüm kritik yetenekler mevcut!")
        return

    st.subheader("🧭 Sana Özel Teknik Gelişim Planı")

    for skill in missing_skills:
        st.markdown(f"### 🔥 {skill.upper()}")

        if skill in ROADMAP_MAP:
            for step in ROADMAP_MAP[skill]:
                st.write(f"• {step}")
        else:
            st.write("• Temel kavramları öğren")
            st.write("• Mini proje geliştir")
            st.write("• Gerçek dünya problemi çöz")

        st.markdown("---")

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

        st.header("📊 Sonuçlar")

        col1, col2, col3 = st.columns(3)
        col1.metric("Hibrit Skor", f"%{final}")
        col2.metric("Semantic Skor", f"%{sem}")
        col3.metric("Keyword Skor", f"%{kw}")

        st.progress(final / 100)

        st.success(f"Karar: {karar(final)}")

        if kw < 40:
            st.warning("⚠ Teknik skill uyumu düşük.")
        if sem < 50:
            st.warning("⚠ CV içeriği ilanla semantik olarak zayıf eşleşiyor.")

        if miss:
            st.warning(f"Eksik yetenekler: {', '.join(miss)}")
        else:
            st.success("Eksik yetenek bulunamadı 🎉")

        st.subheader("📊 Skill Analizi")
        skill_graph(job_sk, cv_sk)

        roadmap(miss)