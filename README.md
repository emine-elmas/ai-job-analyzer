🤖 AI İş Analiz Sistemi
CV ile iş ilanlarını karşılaştıran, eksik yetenekleri tespit eden ve kişiye özel gelişim planı oluşturan yapay zeka destekli web uygulaması.

🚀 Proje Hakkında
Bu proje, kullanıcıların CV’lerini yükleyip iş ilanları ile karşılaştırmasını sağlar.
Sistem; semantic similarity (AI), keyword matching ve skill-based analiz kullanarak uyum skorunu hesaplar ve eksik yetenekleri belirler.
Ayrıca kullanıcıya hangi teknolojileri öğrenmesi gerektiğini adım adım gösteren bir roadmap sunar.

🎯 Özellikler
📄 CV (PDF) yükleme
🧠 AI destekli semantik eşleşme (Sentence Transformers)
🔍 Skill bazlı analiz sistemi
📊 Hibrit uyum skoru (AI + Keyword)
❌ Eksik yetenek tespiti
🧭 Kişisel öğrenme roadmap’i
📈 Skill uyum grafiği
⚡ Streamlit ile interaktif web arayüz

🛠️ Kullanılan Teknolojiler
Python 🐍
Streamlit
SentenceTransformers (AI model)
PyTorch
pdfplumber
Matplotlib
Regex (text processing)

🧠 Sistem Nasıl Çalışır?
Kullanıcı iş ilanını girer
CV PDF yüklenir
CV içeriği analiz edilir
Skill extraction yapılır
AI modeli ile semantik benzerlik hesaplanır
Keyword matching yapılır
Hibrit skor oluşturulur
Eksik yetenekler ve roadmap gösterilir

📊 Örnek Çıktılar
Hibrit Skor: %72
Semantic Skor: %68
Keyword Skor: %75
Eksik Yetenekler: javascript, react
🧭 Roadmap Sistemi
Eksik her yetenek için sistem:
Temel öğrenme adımları
Mini proje önerileri
Gerçek dünya uygulama önerileri
sunarak kişisel gelişim planı oluşturur.

 ▶️ Çalıştırma
Streamlit ile çalışır:
streamlit run app.py

📌 requirements.txt
streamlit
pdfplumber
matplotlib
sentence-transformers
torch

Bu proje eğitim amaçlı geliştirilmiş olup gerçek iş başvurularında adayların kendini geliştirmesine yardımcı bir araçtır.

👩‍💻 Geliştirici
Emine Elmas
📧 emineelmas041@gmail.com
