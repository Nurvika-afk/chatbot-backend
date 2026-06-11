from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json, logging, re, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

INTENTS_PATH = "intents.json"

# ── Konstanta terpusat (tidak ada magic number tersebar) ─────
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
TOPIC_BOOST_VALUE    = float(os.getenv("TOPIC_BOOST_VALUE",    "0.15"))
MULTI_THRESHOLD      = float(os.getenv("MULTI_THRESHOLD",      "0.30"))
MAX_QUESTION_LENGTH  = 500


# ── 1. PREPROCESS ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ── 2. LOAD DATA ──────────────────────────────────────────────
def build_answer_html(answer_obj: dict) -> str:
    """Ubah satu objek answer dari intents.json menjadi HTML."""
    html = f"<b>{answer_obj['text']}</b><br><ol>"
    for poin in answer_obj.get("list", []):
        html += f"<li>{poin}</li>"
    html += "</ol>"

    if answer_obj.get("prosedur"):
        langkah_html = "".join(f"<li>{l}</li>" for l in answer_obj["prosedur"])
        html += (
            "<details>"
            "<summary style='cursor:pointer; color:#1d4ed8; font-weight:bold;'>"
            "📋 Lihat Prosedur Pengajuan Online"
            "</summary>"
            f"<ol style='padding-left:18px; margin-top:8px;'>{langkah_html}</ol>"
            "</details>"
        )

    for idx, sub_item in enumerate(answer_obj.get("sub_prosedur", []), start=1):
        judul        = sub_item.get("judul", f"Kondisi {idx}")
        langkah_html = "".join(
            f"<li style='margin-bottom:4px;'>{l}</li>"
            for l in sub_item.get("langkah", [])
        )
        html += (
            "<details style='margin-bottom:8px; border:1px solid #cbd5e1;"
            " border-radius:8px; padding:8px;'>"
            "<summary style='cursor:pointer; color:#1d4ed8; font-weight:bold;'>"
            f"📂 {idx}. {judul}"
            "</summary>"
            f"<ul style='padding-left:18px; margin-top:8px;'>{langkah_html}</ul>"
            "</details>"
        )

    if answer_obj.get("note"):
        html += f"<br><i>📌 {answer_obj['note']}</i>"

    return html


def load_data():
    if not os.path.exists(INTENTS_PATH):
        raise FileNotFoundError(f"{INTENTS_PATH} tidak ditemukan.")

    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Validasi struktur dasar
    if "data" not in raw or not isinstance(raw["data"], list):
        raise ValueError("intents.json harus memiliki key 'data' berupa list.")

    all_questions: list[str] = []
    all_answers:   list[str] = []

    for i, item in enumerate(raw["data"]):
        if "answer" not in item or "questions" not in item:
            logger.warning(f"Item index {i} tidak memiliki 'answer' atau 'questions', dilewati.")
            continue
        if not isinstance(item["questions"], list) or not item["questions"]:
            logger.warning(f"Item index {i} memiliki 'questions' kosong, dilewati.")
            continue

        answer_text = build_answer_html(item["answer"])

        for question in item["questions"]:
            all_questions.append(preprocess(question))
            all_answers.append(answer_text)

        # Tambahkan keyword sebagai variasi pertanyaan
        if item.get("keywords"):
            keyword_sentence = " ".join(item["keywords"])
            all_questions.append(preprocess(keyword_sentence))
            all_answers.append(answer_text)

    if not all_questions:
        raise ValueError("Tidak ada pertanyaan yang berhasil di-load dari intents.json.")

    logger.info(f"✅ Loaded {len(all_questions)} variasi pertanyaan")
    return all_questions, all_answers


# ── 3. LOAD & BUILD TF-IDF ────────────────────────────────────
try:
    questions, answers = load_data()
except Exception as e:
    logger.critical(f"Gagal load data: {e}")
    raise

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.85,
    sublinear_tf=True,
)
question_vectors = vectorizer.fit_transform(questions)
logger.info("✅ TF-IDF siap.")


# ── Whitelist topik ───────────────────────────────────────────
TOPIK_DILAYANI = [
    "ktp", "kartu tanda penduduk", "e-ktp",
    "kk", "kartu keluarga",
    "akta kelahiran", "akta lahir", "kelahiran",
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
]

def is_topik_dilayani(teks: str) -> bool:
    teks = teks.lower()
    return any(topik in teks for topik in TOPIK_DILAYANI)


# ── Kamus topik & keyword ─────────────────────────────────────
TOPIK_KEYWORDS: dict[str, list[str]] = {
    "kematian":   ["kematian", "meninggal", "wafat", "mati"],
    "kelahiran":  ["kelahiran", "lahir", "bayi"],
    "ktp":        ["ktp", "kartu tanda penduduk", "e-ktp", "ktp-el"],
    "kk":         ["kk", "kartu keluarga"],
    "perkawinan": ["perkawinan", "nikah", "menikah", "kawin"],
    "pindah":     ["pindah", "domisili", "alamat baru"],
    "kia":        ["kia", "kartu identitas anak"],
    "kedatangan": ["datang", "kedatangan", "lapor datang"],
    "cerai":      ["cerai", "akta perceraian", "perceraian"],
}

def topik_dari_pertanyaan(teks: str) -> list[str]:
    """Kembalikan daftar topik yang terdeteksi dalam teks."""
    teks = teks.lower()
    return [
        topik for topik, keywords in TOPIK_KEYWORDS.items()
        if any(k in teks for k in keywords)
    ]


# ── Pra-hitung: topik setiap pertanyaan di dataset ────────────
# Dilakukan sekali saat startup agar tidak dihitung ulang tiap request.
# question_topics[i] = set of topik yang ada di questions[i]
question_topics: list[set[str]] = []
for _q in questions:
    _detected = topik_dari_pertanyaan(_q)
    question_topics.append(set(_detected))

logger.info("✅ Topic index siap.")


def get_topik_kandidat(topik_user: list[str]) -> list[int]:
    """Kembalikan indeks semua pertanyaan di dataset yang topiknya
    bersinggungan dengan topik_user."""
    topik_set = set(topik_user)
    return [i for i, qt in enumerate(question_topics) if qt & topik_set]


def find_best_match(
    raw_question: str,
    similarities,
) -> tuple[int, float]:
    """
    Kembalikan (best_index, best_score).

    Strategi berlapis:

    Layer 1 — Topic-guard (utama):
        Jika topik terdeteksi di pertanyaan user, HANYA pertimbangkan
        kandidat yang topiknya cocok. Ini mencegah "syarat akta perkawinan"
        match ke "syarat akta kematian" meski score cosine-nya lebih tinggi.
        Boost diberikan sekali sebesar TOPIC_BOOST_VALUE.

    Layer 2 — Strict keyword fallback:
        Jika tidak ada kandidat dengan skor >= SIMILARITY_THRESHOLD di layer 1,
        cari pertanyaan di dataset yang mengandung keyword topik user secara
        literal (tanpa memandang cosine score), lalu ambil yang similarity-nya
        paling tinggi di antara mereka.

    Layer 3 — Argmax global:
        Jika topik sama sekali tidak terdeteksi, gunakan argmax similarity biasa.
    """
    topik_user = topik_dari_pertanyaan(raw_question)

    if topik_user:
        # Layer 1: filter kandidat berdasarkan topic index
        kandidat = get_topik_kandidat(topik_user)

        if kandidat:
            best_idx   = max(kandidat, key=lambda i: similarities[i])
            best_score = float(similarities[best_idx])
            boosted    = min(best_score + TOPIC_BOOST_VALUE, 1.0)

            logger.info(
                f"[L1-topic-guard] topik={topik_user}, kandidat={len(kandidat)}, "
                f"idx={best_idx}, raw={best_score:.3f} → boosted={boosted:.3f}"
            )
            return best_idx, boosted

        # Layer 2: topic index kosong → fallback ke keyword literal
        logger.warning(
            f"[L2-keyword-fallback] Tidak ada kandidat untuk topik={topik_user}, "
            "fallback ke keyword literal."
        )
        best_idx   = -1
        best_score = -1.0
        for topik in topik_user:
            for kw in TOPIK_KEYWORDS[topik]:
                for i, q in enumerate(questions):
                    if kw in q and similarities[i] > best_score:
                        best_score = similarities[i]
                        best_idx   = i

        if best_idx >= 0:
            boosted = min(best_score + TOPIC_BOOST_VALUE, 1.0)
            logger.info(f"[L2] idx={best_idx}, boosted={boosted:.3f}")
            return best_idx, boosted

    # Layer 3: tidak ada topik → argmax global
    best_idx   = int(similarities.argmax())
    best_score = float(similarities[best_idx])
    logger.info(f"[L3-global] idx={best_idx}, score={best_score:.3f}")
    return best_idx, best_score


# ── 4. REQUEST & RESPONSE MODEL ───────────────────────────────
class ChatRequest(BaseModel):
    question: str = ""
    message:  str = ""

class ChatResponse(BaseModel):
    answer:     str
    confidence: float


# ── 5. ENDPOINT CHAT ──────────────────────────────────────────
KATA_GANDA = {"dan", "serta", "juga", "sekaligus", "dengan", "atau"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    raw_question = (request.question or request.message).strip()

    if not raw_question:
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")
    if len(raw_question) > MAX_QUESTION_LENGTH:
        raise HTTPException(status_code=400, detail="Pertanyaan terlalu panjang.")

    # Whitelist topik
    if not is_topik_dilayani(raw_question):
        fallback = (
            "Maaf, pertanyaan Anda di luar layanan yang tersedia. 🙏<br>"
            "Saya hanya melayani informasi seputar layanan "
            "Disdukcapil Kota Semarang seperti KTP, KK, Akta Kelahiran, "
            "Akta Kematian, Akta Perkawinan, dan layanan kependudukan lainnya."
        )
        return ChatResponse(answer=fallback, confidence=0.0)

    user_vector  = vectorizer.transform([preprocess(raw_question)])
    similarities = cosine_similarity(user_vector, question_vectors)[0]

    # DEBUG: top-5 kandidat
    top5 = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:5]
    for idx, score in top5:
        logger.info(f"  [{score:.3f}] {questions[idx][:60]}")

    best_index, best_score = find_best_match(raw_question, similarities)

    # Threshold check
    if best_score < SIMILARITY_THRESHOLD:
        fallback = (
            "Maaf, saya belum bisa menjawab pertanyaan tersebut. 🙏 "
            "Silakan hubungi Disdukcapil Kota Semarang secara langsung "
            "atau coba pertanyaan lain."
        )
        return ChatResponse(answer=fallback, confidence=best_score)

    # Deteksi multi-topik
    tokens_lower  = set(raw_question.lower().split())
    ada_kata_ganda = bool(tokens_lower & KATA_GANDA)
    topik_user     = topik_dari_pertanyaan(raw_question)

    logger.info(f"Topik terdeteksi: {topik_user}")

    if ada_kata_ganda and len(topik_user) >= 2:
        seen_answers: list[str] = []

        for topik in topik_user[:2]:
            keywords      = TOPIK_KEYWORDS[topik]
            best_t_score  = -1.0
            best_t_index  = -1

            for i, q in enumerate(questions):
                if any(k in q for k in keywords) and similarities[i] > best_t_score:
                    best_t_score = similarities[i]
                    best_t_index = i

            if best_t_index >= 0 and answers[best_t_index] not in seen_answers:
                seen_answers.append(answers[best_t_index])

        if len(seen_answers) >= 2:
            combined = (
                "<b>Saya menemukan informasi untuk beberapa topik:</b><br><br>"
                f"<b>📌 Topik 1:</b><br>{seen_answers[0]}"
                "<br><hr style='border:1px dashed #e5e7eb; margin:10px 0;'>"
                f"<b>📌 Topik 2:</b><br>{seen_answers[1]}"
            )
            return ChatResponse(answer=combined, confidence=best_score)

    return ChatResponse(answer=answers[best_index], confidence=best_score)


# ── 6. HEALTH CHECK ───────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {
        "status":           "ok",
        "total_questions":  len(questions),
        "threshold":        SIMILARITY_THRESHOLD,
        "topic_boost":      TOPIC_BOOST_VALUE,
    }


# ── 7. DEBUG: TF-IDF ─────────────────────────────────────────
@app.get("/tfidf")
async def get_tfidf(question: str):
    user_vector   = vectorizer.transform([preprocess(question)])
    feature_names = vectorizer.get_feature_names_out()

    tfidf_scores = {
        feature_names[idx]: round(float(score), 4)
        for idx, score in zip(user_vector.indices, user_vector.data)
    }
    tfidf_sorted = dict(sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True))

    similarities         = cosine_similarity(user_vector, question_vectors)[0]
    best_index, boosted  = find_best_match(question, similarities)

    return {
        "question":         question,
        "preprocessed":     preprocess(question),
        "tfidf_scores":     tfidf_sorted,
        "total_features":   len(feature_names),
        "cosine_score":     round(float(similarities[best_index]), 4),
        "boosted_score":    round(boosted, 4),
        "threshold":        SIMILARITY_THRESHOLD,
        "matched_question": questions[best_index],
        "akan_terjawab":    boosted >= SIMILARITY_THRESHOLD,
    }


# ── 8. DEBUG: VOCABULARY ─────────────────────────────────────
@app.get("/tfidf/vocabulary")
async def get_vocabulary():
    feature_names = vectorizer.get_feature_names_out()
    return {
        "total_vocabulary": len(feature_names),
        "vocabulary":       list(feature_names),
    }
