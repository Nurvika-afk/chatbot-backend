from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import json, logging, re, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if os.path.exists("dist"):
    app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse("dist/index.html")

# ── Spellchecker Setup ───────────────────────────────────────
spell = SpellChecker(language=None)

kamus_dukcapil = [
    # Dokumen utama
    "ktp", "ektp", "kartu", "tanda", "penduduk",
    "kk", "keluarga",
    "akta", "kelahiran", "lahir",
    "kematian", "meninggal",
    "perkawinan", "nikah", "menikah",
    "perceraian", "cerai",
    "kia", "identitas", "anak",
    # Layanan
    "pindah", "domisili", "surat", "skpwni",
    "kedatangan", "lapor",
    "kontrak", "numpang",
    # Institusi
    "disdukcapil", "dukcapil", "sidnok", "semarang",
    "administrasi", "kependudukan", "pencatatan", "sipil",
    # Operasional
    "jam", "buka", "tutup", "operasional", "kantor",
    "hari", "pelayanan",
    # Digital
    "ikd", "digital", "password", "reset", "email", "nomor",
    # Sapaan
    "halo", "hai", "terima", "makasih", "selamat",
    "tolong", "bantu",
    # Kata kerja umum
    "syarat", "cara", "prosedur", "biaya", "daftar",
    "buat", "urus", "ganti", "ubah", "fotokopi",
    "rumah", "pernyataan", "sendiri",
    # Perubahan data KK
    "perubahan", "biodata", "data", "anggota", "kepala",
    "alamat", "pekerjaan", "pendidikan", "status", "tambah",
    # Kata sambung / stopwords umum
    "atau", "dan", "dengan", "yang", "untuk", "dari", "pada",
    "akan", "adalah", "karena", "agar", "supaya", "jika", "kalau",
    "saya", "anda", "kami", "kita", "mereka", "ingin", "mau",
    "bisa", "dapat", "harus", "perlu", "sudah", "belum", "tidak",
    "apa", "apakah", "bagaimana", "dimana", "kapan", "siapa",
    "ini", "itu", "ada", "saja", "juga", "serta", "atas", "oleh",
    "tentang", "seputar", "mengenai", "terkait", "soal",
    "bagaimana", "begini", "begitu", "lalu", "kemudian", "setelah",
    "sebelum", "saat", "ketika", "tahun", "bulan", "tanggal",
    "nama", "anak", "istri", "suami", "orangtua", "orang", "tua",
    "pria", "wanita", "laki", "perempuan", "umur", "usia",
    "tanya", "bertanya", "pertanyaan", "informasi", "info",
]
spell.word_frequency.load_words(kamus_dukcapil)

def koreksi_typo(text: str) -> str:
    words = text.split()
    corrected = []
    for word in words:
        if len(word) > 3:
            correction = spell.correction(word)
            corrected.append(correction if correction else word)
        else:
            corrected.append(word)
    result = " ".join(corrected)
    if result != text:
        logger.info(f"Koreksi typo: '{text}' → '{result}'")
    return result

INTENTS_PATH = "intents.json"

def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def load_data():
    if not os.path.exists(INTENTS_PATH):
        raise FileNotFoundError(f"{INTENTS_PATH} tidak ditemukan.")

    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_questions = []
    all_answers   = []

    for item in data["data"]:
        answer_obj = item["answer"]

        answer_text = f"<b>{answer_obj['text']}</b><br><ol>"
        for poin in answer_obj.get("list", []):
            answer_text += f"<li>{poin}</li>"
        answer_text += "</ol>"

        prosedur = answer_obj.get("prosedur", [])
        if prosedur:
            langkah_html = "".join(f"<li>{langkah}</li>" for langkah in prosedur)
            answer_text += (
                "<details>"
                "<summary style='cursor:pointer; color:#1d4ed8; font-weight:bold;'>"
                "📋 Lihat Prosedur Pengajuan Online"
                "</summary>"
                f"<ol style='padding-left:18px; margin-top:8px;'>{langkah_html}</ol>"
                "</details>"
            )

        sub_prosedur = answer_obj.get("sub_prosedur", [])
        if sub_prosedur:
            for index, sub_item in enumerate(sub_prosedur, start=1):
                judul   = sub_item.get("judul", f"Kondisi {index}")
                langkah = sub_item.get("langkah", [])
                langkah_html = "".join(
                    f"<li style='margin-bottom:4px;'>{l}</li>" for l in langkah
                )
                answer_text += (
                    "<details style='margin-bottom:8px; border:1px solid #cbd5e1; border-radius:8px; padding:8px;'>"
                    "<summary style='cursor:pointer; color:#1d4ed8; font-weight:bold;'>"
                    f"📂 {index}. {judul}"
                    "</summary>"
                    f"<ul style='padding-left:18px; margin-top:8px;'>{langkah_html}</ul>"
                    "</details>"
                )

        if answer_obj.get("note"):
            answer_text += f"<br><i>📌 {answer_obj['note']}</i>"

        for question in item["questions"]:
            all_questions.append(preprocess(question))
            all_answers.append(answer_text)

        if item.get("keywords"):
            keyword_sentence = " ".join(item["keywords"])
            all_questions.append(preprocess(keyword_sentence))
            all_answers.append(answer_text)

    logger.info(f"✅ Loaded {len(all_questions)} variasi pertanyaan")
    return all_questions, all_answers

try:
    questions, answers = load_data()
except Exception as e:
    logger.critical(f"Gagal load data: {e}")
    raise

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.85,
    sublinear_tf=True
)
question_vectors = vectorizer.fit_transform(questions)
logger.info("✅ TF-IDF siap.")

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))

# ── Daftar topik yang DILAYANI Disdukcapil ──────────────────
TOPIK_DILAYANI = [
    "ktp", "kartu tanda penduduk", "e-ktp",
    "kk", "kartu keluarga",
    "akta", "akta kelahiran", "akta lahir", "kelahiran",
    "akta kematian", "kematian", "meninggal",
    "akta perkawinan", "nikah", "menikah", "perkawinan",
    "akta perceraian", "cerai", "perceraian",
    "kia", "kartu identitas anak",
    "pindah", "domisili", "surat pindah", "skpwni",
    "kedatangan", "lapor datang",
    "disdukcapil", "dukcapil", "sidnok", "semarang",
    "administrasi", "kependudukan", "pencatatan sipil",
    "halo", "hai", "terima kasih", "makasih",
    "selamat pagi", "selamat siang", "selamat malam",
    "oke", "baik", "bye", "tolong", "bantu",
    "jam", "buka", "tutup", "operasional", "kantor", "hari", "pelayanan",
    "rumah sendiri", "surat pernyataan rumah sendiri",
    "kontrak", "surat pernyataan kontrak rumah",
    "numpang kk", "surat pernyataan numpang kk",
    "si d'nok", "ikd", "identitas kependudukan digital",
    "reset password", "lupa password", "ubah email", "ubah nomor",
    "ubah data kk", "perubahan data kk", "perubahan biodata",
    "ubah kk", "perubahan kk", "ganti data kk", "biodata kk",
    "pekerjaan", "pendidikan", "perubahan pekerjaan", "perubahan pendidikan",
    "ganti pekerjaan", "ganti pendidikan", "ubah pekerjaan", "ubah pendidikan",
    "perubahan nama", "ganti nama", "ubah nama", "salah nama",
    "perubahan alamat", "ganti alamat", "ubah alamat",
    "tambah anggota keluarga", "tambah anggota", "anggota keluarga baru",
    "perubahan status", "ganti status", "ubah status",
    "status perkawinan", "status kawin", "status cerai", "status janda", "status duda",
]

# ── Daftar topik yang BUKAN layanan Disdukcapil ─────────────
# Singkatan / kata ini langsung ditolak sebelum masuk TF-IDF
TOPIK_DILARANG = [
    # Kepolisian / hukum
    "skck", "surat catatan kepolisian", "tilang",
    "sim", "stnk", "bpkb",
    "polisi", "polsek", "polres", "laporan polisi",
    # Kesehatan / BPJS
    "bpjs", "jkn", "puskesmas", "rumah sakit",
    "klinik", "vaksin", "imunisasi",
    "kartu sehat", "kis", "faskes",
    # Pajak / keuangan
    "npwp", "pajak", "spt", "bphtb", "pbb",
    "bank", "rekening", "atm", "transfer",
    # Pendidikan / sekolah
    "ijazah", "skl", "surat kelulusan", "rapor",
    "sekolah", "kampus", "universitas",
    # Imigrasi / dokumen perjalanan
    "paspor", "visa", "imigrasi", "kitas", "kitap",
    # Bantuan sosial
    "pkh", "blt", "bansos", "bantuan sosial", "kartu prakerja",
    # Perizinan usaha / bangunan
    "bpom", "sertifikat halal", "izin usaha", "siup", "nib",
    "imb", "imt", "izin mendirikan bangunan",
    # Pertanahan
    "akta tanah", "sertifikat tanah", "bpn", "shm",
    # Lain-lain
    "kartu pos",
]

def is_topik_dilarang(teks: str) -> bool:
    """Cek apakah pertanyaan mengandung topik yang bukan layanan Disdukcapil."""
    teks = teks.lower()
    return any(topik in teks for topik in TOPIK_DILARANG)

def is_topik_dilayani(teks: str) -> bool:
    teks = teks.lower()
    return any(topik in teks for topik in TOPIK_DILAYANI)

topik_keywords = {
    "kematian": ["kematian", "meninggal", "wafat", "mati"],
    "kelahiran": ["kelahiran", "lahir", "bayi"],
    "ktp": ["ktp", "kartu tanda penduduk", "e-ktp", "ktp-el"],
    "kk": ["kk", "kartu keluarga"],
    "perkawinan": ["perkawinan", "nikah", "menikah", "kawin"],
    "pindah": ["pindah", "domisili", "alamat baru"],
    "kia": ["kia", "kartu identitas anak"],
    "kedatangan": ["datang", "kedatangan", "lapor datang"],
    "cerai": ["cerai", "akta perceraian", "perceraian"],
    "jam_operasional": ["jam", "buka", "tutup", "operasional", "kantor", "hari", "pelayanan"],
    "surat_pernyataan": ["rumah sendiri", "surat pernyataan rumah sendiri", "siapa", "membuat", "menandatangani","kontrak", "surat pernyataan kontrak rumah","numpang kk", "surat pernyataan numpang kk"],
    "si_d'nok": ["si d'nok", "ubah email", "ubah password", "ikd", "identitas kependudukan digital", "reset password", "lupa password", "ubah email", "ubah nomor"],
    "biodata_nama": [
        "perubahan nama", "ganti nama", "ubah nama", "salah nama",
        "perbaikan nama", "nama salah", "perbaikan data nama",
    ],
    "biodata_status_kawin": [
        "perubahan status perkawinan", "ganti status perkawinan", "ubah status perkawinan",
        "status kawin", "status cerai", "ubah status", "ganti status",
        "perubahan status", "status janda", "status duda",
    ],
    "biodata_pendidikan_pekerjaan": [
        "perubahan pendidikan", "ganti pendidikan", "ubah pendidikan", "pendidikan",
        "perubahan pekerjaan", "ganti pekerjaan", "ubah pekerjaan", "pekerjaan",
        "update pendidikan", "update pekerjaan", "ubah data karena perubahan pendidikan atau pekerjaan",
    ],
    "biodata_pindah_alamat": [
        "pindah alamat", "ubah data karena pindah alamat", "perubahan alamat",
        "ganti alamat", "ubah alamat", "pindah domisili", "pindah kelurahan",
        "pindah kecamatan", "pindah dalam kota", "pindah antar kelurahan",
        "pindah keluar", "surat pindah", "domisili",
    ],
}

def topik_dari_pertanyaan(teks: str) -> list:
    hasil = []
    for topik, keywords in topik_keywords.items():
        if any(k in teks for k in keywords):
            hasil.append(topik)
    return hasil

def boost_score(raw_question: str, base_score: float, best_index: int) -> float:
    topik_user = topik_dari_pertanyaan(raw_question.lower())
    if not topik_user:
        return base_score
    pertanyaan_terbaik = questions[best_index].lower()
    boost = 0.0
    for topik in topik_user:
        keywords_topik = topik_keywords.get(topik, [])
        if any(k in pertanyaan_terbaik for k in keywords_topik):
            boost += 0.15
    boosted = min(base_score + boost, 1.0)
    logger.info(f"Score boost: {base_score:.3f} → {boosted:.3f} (topik: {topik_user})")
    return boosted

def get_best_index_for_topik(topik_user: list, similarities) -> int:
    best_score = -1
    best_idx   = -1
    for topik in topik_user:
        keywords_topik = topik_keywords.get(topik, [])
        for i, q in enumerate(questions):
            if any(k in q for k in keywords_topik):
                if similarities[i] > best_score:
                    best_score = similarities[i]
                    best_idx   = i
    return best_idx

# ============================================================
# AMBIGUOUS TOPIC HANDLER (Clarification Options)
# ============================================================
AMBIGUOUS_TOPICS = {
    "akta": {
        "trigger": ["akta"],
        "specific": [
            "akta kelahiran", "akta lahir",
            "akta kematian", "kematian", "meninggal",
            "akta perkawinan", "akta nikah", "nikah", "menikah", "kawin",
            "akta perceraian", "akta cerai", "cerai", "perceraian",
        ],
        "question": "Anda ingin bertanya tentang akta apa? Silakan pilih salah satu di bawah ini 👇",
        "options": [
            "Syarat akta kelahiran",
            "Syarat akta kematian",
            "Syarat akta perkawinan",
            "Syarat akta perceraian",
        ],
    },
    "ubah_kk": {
        "trigger": [
            "ubah data kk", "perubahan data kk", "ubah kk",
            "perubahan kk", "ganti data kk", "perubahan biodata kk",
            "ubah biodata kk", "ganti biodata kk", "perbaikan data kk",
            "koreksi data kk",
        ],
        "specific": [],
        "question": "Perubahan data KK apa yang Anda maksud? Silakan pilih salah satu di bawah ini 👇",
        "options": [
            "Ubah data karena perubahan nama",
            "Ubah data karena perubahan status perkawinan",
            "Ubah data karena perubahan pendidikan atau pekerjaan",
            "Ubah data karena pindah alamat",
        ],
    },
}

def cek_topik_ambigu(raw_question: str):
    teks = raw_question.lower()
    for key, cfg in AMBIGUOUS_TOPICS.items():
        if cfg["specific"] and any(s in teks for s in cfg["specific"]):
            continue
        if any(t in teks for t in cfg["trigger"]):
            logger.info(f"Topik ambigu terdeteksi: '{key}' dari pertanyaan '{raw_question}'")
            return cfg
    return None


class ChatRequest(BaseModel):
    question: str = ""
    message: str = ""

class ChatResponse(BaseModel):
    answer: str
    reply: str
    confidence: float
    options: Optional[List[str]] = None

MULTI_THRESHOLD = 0.30

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    raw_question = (request.question or request.message).strip()

    if not raw_question:
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")

    if len(raw_question) > 500:
        raise HTTPException(status_code=400, detail="Pertanyaan terlalu panjang.")

    # ✅ Tolak topik yang jelas bukan layanan Disdukcapil (SKCK, BPJS, SIM, dst.)
    # PENTING: dicek dari teks ASLI user sebelum koreksi typo,
    # agar singkatan seperti "skck" tidak diubah menjadi kata lain oleh SpellChecker.
    if is_topik_dilarang(raw_question):
        fallback = (
            "Maaf, pertanyaan Anda berkaitan dengan layanan di luar kewenangan "
            "Disdukcapil Kota Semarang. 🙏<br><br>"
            "Saya hanya melayani informasi seputar:<br>"
            "<ul>"
            "<li>📋 KTP / e-KTP</li>"
            "<li>📄 Kartu Keluarga (KK)</li>"
            "<li>👶 Akta Kelahiran</li>"
            "<li>💍 Akta Perkawinan &amp; Perceraian</li>"
            "<li>⚰️ Akta Kematian</li>"
            "<li>🏠 Pindah Domisili &amp; Surat Kependudukan</li>"
            "<li>📱 Identitas Kependudukan Digital (IKD)</li>"
            "</ul>"
            "Untuk layanan lain, silakan hubungi instansi terkait. 🙏"
        )
        return ChatResponse(answer=fallback, reply=fallback, confidence=0.0, options=None)

    # ✅ Koreksi typo — dilakukan SETELAH filter topik dilarang
    raw_question = koreksi_typo(raw_question)

    # ✅ Cek apakah topik termasuk layanan Disdukcapil
    if not is_topik_dilayani(raw_question):
        fallback = (
            "Maaf, pertanyaan Anda di luar layanan yang tersedia. 🙏<br>"
            "Saya hanya melayani informasi seputar layanan "
            "Disdukcapil Kota Semarang seperti KTP, KK, Akta Kelahiran, "
            "Akta Kematian, Akta Perkawinan, dan layanan kependudukan lainnya."
        )
        return ChatResponse(answer=fallback, reply=fallback, confidence=0.0, options=None)

    # ✅ Cek apakah topik masih ambigu / terlalu umum
    ambigu = cek_topik_ambigu(raw_question)
    if ambigu:
        return ChatResponse(
            answer=ambigu["question"],
            reply=ambigu["question"],
            confidence=1.0,
            options=ambigu["options"],
        )

    user_question = preprocess(raw_question)
    user_vector   = vectorizer.transform([user_question])
    similarities  = cosine_similarity(user_vector, question_vectors)[0]

    best_score = float(similarities.max())
    best_index = int(similarities.argmax())

    topik_user = topik_dari_pertanyaan(raw_question.lower())

    if topik_user:
        topik_index = get_best_index_for_topik(topik_user, similarities)
        if topik_index >= 0:
            best_index = topik_index
            best_score = float(similarities[best_index]) + 0.15
            best_score = min(best_score, 1.0)
            logger.info(f"Topik override: index={best_index}, score={best_score:.3f}")

    best_score = boost_score(raw_question, best_score, best_index)

    top5 = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:5]
    for idx, score in top5:
        logger.info(f"  [{score:.3f}] {questions[idx][:60]}")

    if best_score < SIMILARITY_THRESHOLD:
        fallback = (
            "Maaf, saya belum bisa menjawab pertanyaan tersebut. 🙏 "
            "Silakan hubungi Disdukcapil Kota Semarang secara langsung "
            "atau coba pertanyaan lain."
        )
        return ChatResponse(answer=fallback, reply=fallback, confidence=best_score, options=None)

    kata_ganda = ["dan", "serta", "juga", "sekaligus", "dengan", "atau"]
    ada_kata_ganda = any(f" {k} " in f" {raw_question.lower()} " for k in kata_ganda)
    topik_user = topik_dari_pertanyaan(raw_question.lower())
    logger.info(f"Topik terdeteksi: {topik_user}")
    is_multi = ada_kata_ganda and len(topik_user) >= 2

    if is_multi:
        seen_answers = []
        for topik in topik_user[:2]:
            keywords_topik = topik_keywords.get(topik, [])
            best_topik_score = -1
            best_topik_index = -1
            for i, q in enumerate(questions):
                if any(k in q for k in keywords_topik):
                    if similarities[i] > best_topik_score:
                        best_topik_score = similarities[i]
                        best_topik_index = i
            if best_topik_index >= 0 and answers[best_topik_index] not in seen_answers:
                seen_answers.append(answers[best_topik_index])

        if len(seen_answers) >= 2:
            combined = (
                "<b>Saya menemukan informasi untuk beberapa topik:</b><br><br>"
                f"<b>📌 Topik 1:</b><br>{seen_answers[0]}"
                "<br><hr style='border:1px dashed #e5e7eb; margin:10px 0;'>"
                f"<b>📌 Topik 2:</b><br>{seen_answers[1]}"
            )
            return ChatResponse(answer=combined, reply=combined, confidence=best_score, options=None)

    return ChatResponse(
        answer=answers[best_index],
        reply=answers[best_index],
        confidence=best_score,
        options=None,
    )

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "total_questions": len(questions),
        "threshold": SIMILARITY_THRESHOLD,
    }

@app.get("/tfidf")
async def get_tfidf(question: str):
    user_question = preprocess(question)
    user_vector   = vectorizer.transform([user_question])
    feature_names = vectorizer.get_feature_names_out()

    tfidf_scores = {}
    for idx, score in zip(user_vector.indices, user_vector.data):
        tfidf_scores[feature_names[idx]] = round(float(score), 4)

    tfidf_sorted = dict(
        sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    )

    similarities  = cosine_similarity(user_vector, question_vectors)[0]
    best_score    = float(similarities.max())
    best_index    = int(similarities.argmax())
    best_question = questions[best_index]
    boosted_score = boost_score(question, best_score, best_index)

    return {
        "question"        : question,
        "preprocessed"    : user_question,
        "tfidf_scores"    : tfidf_sorted,
        "total_features"  : len(feature_names),
        "cosine_score"    : round(best_score, 4),
        "boosted_score"   : round(boosted_score, 4),
        "threshold"       : SIMILARITY_THRESHOLD,
        "matched_question": best_question,
        "akan_terjawab"   : boosted_score >= SIMILARITY_THRESHOLD
    }

@app.get("/tfidf/vocabulary")
async def get_vocabulary():
    feature_names = vectorizer.get_feature_names_out()
    return {
        "total_vocabulary": len(feature_names),
        "vocabulary"      : list(feature_names)
    }
