"""
╔══════════════════════════════════════════════════════════════╗
║         EVALUASI THRESHOLD - BESTI Disdukcapil              ║
║  Jalankan setiap kali menambah intent baru ke intents.json  ║
╚══════════════════════════════════════════════════════════════╝

Cara pakai:
    python evaluasi.py
    python evaluasi.py --threshold 0.45   (uji threshold tertentu)
    python evaluasi.py --verbose          (tampilkan semua detail)
"""

import json
import re
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ══════════════════════════════════════════════════════════════
# KONFIGURASI — edit bagian ini sesuai kebutuhan
# ══════════════════════════════════════════════════════════════

INTENTS_FILE = "intents.json"

# Pertanyaan uji — tambah terus seiring intent baru ditambahkan
# Format: ("pertanyaan user", "ID_YANG_DIHARAPKAN" atau None)
# None = pertanyaan di luar topik, harusnya TIDAK dijawab
TEST_CASES = [

    # ── KTP ────────────────────────────────────────────────────
    ("mau bikin ktp",                          "KTP-001"),
    ("buat ktp pertama kali",                  "KTP-001"),
    ("syarat perekaman ktp el",                "KTP-001"),
    ("ktp saya hilang",                        "KTP-002"),
    ("kartu tanda penduduk rusak",             "KTP-002"),
    ("kartu tanda penduduk hilang",            "KTP-002"),

    # ── Kartu Keluarga ─────────────────────────────────────────
    ("syarat membuat kk baru",                 "KK-001"),
    ("buat kartu keluarga setelah menikah",    "KK-001"),
    ("tambah anggota keluarga di kk",          "KK-002"),
    ("ubah data di kartu keluarga",            "KK-002"),

    # ── Akta Kelahiran ─────────────────────────────────────────
    ("buat akta kelahiran bayi",               "AKTA-001"),
    ("cara daftar akta lahir",                 "AKTA-001"),

    # ── Informasi Umum ─────────────────────────────────────────
    ("jam buka disdukcapil",                   "INFO-001"),
    ("disdukcapil buka hari sabtu tidak",      "INFO-001"),

    # ── Perpindahan Keluar ─────────────────────────────────────
    ("mau pindah keluar semarang",             "PINDAH-001"),
    ("surat pindah ke kota lain",              "PINDAH-001"),
    ("pindah antar kelurahan semarang",        "PINDAH-002"),
    ("pindah kecamatan dalam kota",            "PINDAH-002"),

    # ── Kedatangan ─────────────────────────────────────────────
    ("baru pindah ke semarang",                "DATANG-001"),
    ("cara lapor kedatangan di semarang",      "DATANG-001"),
    ("orang asing mau tinggal semarang",       "DATANG-002"),
    ("wna pindah ke semarang syarat apa",      "DATANG-002"),

    # ── Perubahan Biodata KK ───────────────────────────────────
    ("nama saya salah di kk",                  "BIODATA-001"),
    ("ganti nama di kartu keluarga",           "BIODATA-001"),
    ("ubah status kawin di kk",                "BIODATA-002"),
    ("sudah cerai mau ubah status di kk",      "BIODATA-002"),
    ("ganti data pekerjaan di kk",             "BIODATA-003"),
    ("update pendidikan di kartu keluarga",    "BIODATA-003"),

    # ── KIA ────────────────────────────────────────────────────
    ("buat kartu identitas anak",              "KIA-001"),
    ("cara daftar kia untuk anak",             "KIA-001"),
    ("kia anak saya hilang",                   "KIA-002"),
    ("kartu identitas anak rusak",             "KIA-002"),

    # ── Akta Nikah/Cerai Non-Muslim ────────────────────────────
    ("akta nikah non muslim",                  "NIKAH-001"),
    ("daftar pernikahan kristen di dukcapil",  "NIKAH-001"),
    ("cerai non muslim dokumen apa",           "CERAI-001"),
    ("cara urus akta cerai di dukcapil",       "CERAI-001"),

    # ── Akta Kematian ──────────────────────────────────────────
    ("cara buat akta kematian",                "MATI-001"),
    ("surat kematian nenek syaratnya apa",     "MATI-001"),
    ("almarhum meninggal di luar kota",        "MATI-002"),
    ("akta kematian beda domisili",            "MATI-002"),

    # ── DI LUAR TOPIK (harusnya TIDAK dijawab / skor rendah) ──
    ("cuaca hari ini",                         None),
    ("harga beras naik",                       None),
    ("rekomendasi restoran enak",              None),
    ("jadwal kereta api",                      None),
    ("cara daftar bpjs",                       None),
    ("info tagihan listrik pln",               None),
    ("cara beli tiket pesawat",                None),
]


# ══════════════════════════════════════════════════════════════
# FUNGSI UTAMA
# ══════════════════════════════════════════════════════════════

def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_intents(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_questions = []
    all_labels    = []

    for item in data["data"]:
        for q in item["questions"]:
            all_questions.append(preprocess(q))
            all_labels.append(item["id"])

    return data, all_questions, all_labels


def build_vectorizer(questions):
    vectorizer       = TfidfVectorizer(ngram_range=(1, 2))
    question_vectors = vectorizer.fit_transform(questions)
    return vectorizer, question_vectors


def run_evaluation(vectorizer, question_vectors, all_labels, threshold, verbose):

    scores_cocok  = []
    scores_diluar = []
    results       = []

    for user_q, expected_id in TEST_CASES:
        user_vec     = vectorizer.transform([preprocess(user_q)])
        similarities = cosine_similarity(user_vec, question_vectors)
        best_score   = float(similarities.max())
        best_index   = int(similarities.argmax())
        best_label   = all_labels[best_index]

        if expected_id is None:
            # Pertanyaan di luar topik
            dijawab = best_score >= threshold
            status  = "⚠️  SALAH TANGKAP" if dijawab else "✅ AMAN"
            benar   = not dijawab
            scores_diluar.append(best_score)
        else:
            # Pertanyaan dalam topik
            label_cocok  = best_label == expected_id
            skor_cukup   = best_score >= threshold
            if label_cocok and skor_cukup:
                status = "✅ BENAR"
                benar  = True
            elif not skor_cukup:
                status = "❌ DITOLAK (skor terlalu rendah)"
                benar  = False
            else:
                status = f"❌ SALAH → dapat {best_label}"
                benar  = False
            scores_cocok.append((best_score, label_cocok))

        results.append({
            "pertanyaan"  : user_q,
            "expected"    : expected_id,
            "hasil"       : best_label,
            "skor"        : best_score,
            "status"      : status,
            "benar"       : benar,
        })

    return results, scores_cocok, scores_diluar


def print_report(results, scores_cocok, scores_diluar, threshold, data, all_questions, verbose):

    total  = len(results)
    benar  = sum(1 for r in results if r["benar"])
    akurasi = benar / total * 100

    # ── Header ────────────────────────────────────────────────
    print("\n" + "╔" + "═" * 88 + "╗")
    print("║" + "  LAPORAN EVALUASI THRESHOLD — BESTI Disdukcapil".center(88) + "║")
    print("╚" + "═" * 88 + "╝")
    print(f"  File intents : {INTENTS_FILE}")
    print(f"  Total intent : {len(data['data'])}")
    print(f"  Total var. Q : {len(all_questions)}")
    print(f"  Threshold    : {threshold}")

    # ── Tabel hasil ───────────────────────────────────────────
    if verbose:
        print(f"\n{'PERTANYAAN USER':<42} {'EKSPEKTASI':<12} {'HASIL':<12} {'SKOR':>6}  STATUS")
        print("─" * 95)
        for r in results:
            exp = str(r["expected"]) if r["expected"] else "DILUAR"
            print(f"  {r['pertanyaan']:<40} {exp:<12} {r['hasil']:<12} {r['skor']:>6.3f}  {r['status']}")

    # ── Hanya yang gagal (mode ringkas) ───────────────────────
    else:
        gagal = [r for r in results if not r["benar"]]
        if gagal:
            print(f"\n  ❌ PERTANYAAN YANG GAGAL:")
            print(f"  {'PERTANYAAN USER':<42} {'EKSPEKTASI':<12} {'HASIL':<12} {'SKOR':>6}")
            print("  " + "─" * 80)
            for r in gagal:
                exp = str(r["expected"]) if r["expected"] else "DILUAR"
                print(f"  {r['pertanyaan']:<42} {exp:<12} {r['hasil']:<12} {r['skor']:>6.3f}  {r['status']}")
        else:
            print("\n  ✅ Semua pertanyaan uji berhasil!")

    # ── Statistik skor ────────────────────────────────────────
    skor_hanya_nilai = [s for s, _ in scores_cocok]

    print("\n" + "═" * 90)
    print(f"  Akurasi                           : {benar}/{total} ({akurasi:.1f}%)")
    print(f"  Skor minimum pertanyaan COCOK     : {min(skor_hanya_nilai):.3f}")
    print(f"  Skor rata-rata pertanyaan COCOK   : {sum(skor_hanya_nilai)/len(skor_hanya_nilai):.3f}")
    print(f"  Skor maksimum pertanyaan DI LUAR  : {max(scores_diluar):.3f}")
    print(f"  Skor rata-rata pertanyaan DI LUAR : {sum(scores_diluar)/len(scores_diluar):.3f}")

    # ── Rekomendasi threshold ─────────────────────────────────
    min_cocok  = min(skor_hanya_nilai)
    max_diluar = max(scores_diluar)
    rekomendasi = (min_cocok + max_diluar) / 2

    print("\n" + "─" * 90)
    if min_cocok > max_diluar:
        print(f"  ✅ Ada celah aman antara pertanyaan COCOK dan DI LUAR TOPIK.")
        print(f"     Range aman  : {max_diluar:.3f} → {min_cocok:.3f}")
        print(f"     💡 REKOMENDASI THRESHOLD : {rekomendasi:.3f}")
    else:
        print(f"  ⚠️  ADA OVERLAP — pertanyaan di luar topik memiliki skor lebih tinggi")
        print(f"     dari pertanyaan yang seharusnya cocok.")
        print(f"     Tambah lebih banyak variasi pertanyaan ke intent yang relevan.")
        print(f"     💡 REKOMENDASI THRESHOLD SEMENTARA : {rekomendasi:.3f} (perlu perbaikan data)")

    # ── Visualisasi distribusi skor ───────────────────────────
    print("\n  Distribusi skor (● = cocok, ○ = di luar topik):\n")
    semua = [(s, "●") for s, _ in scores_cocok] + [(s, "○") for s in scores_diluar]
    semua.sort()
    batas_kiri  = 0.0
    batas_kanan = 1.0
    lebar       = 60

    skala = ""
    bar   = ""
    for skor, simbol in semua:
        pos = int((skor - batas_kiri) / (batas_kanan - batas_kiri) * lebar)
        bar += " " * (pos - len(bar)) + simbol if pos >= len(bar) else simbol

    threshold_pos = int((threshold - batas_kiri) / (batas_kanan - batas_kiri) * lebar)
    bar_list = list(bar.ljust(lebar + 1))
    if threshold_pos < len(bar_list):
        bar_list[threshold_pos] = "│"
    bar = "".join(bar_list)

    print(f"  0.0{' ' * (threshold_pos - 2)}T{' ' * (lebar - threshold_pos - 1)}1.0")
    print(f"   {bar}")
    print(f"  ○ = di luar topik   ● = cocok   │ = threshold ({threshold})\n")

    # ── Saran perbaikan ───────────────────────────────────────
    gagal_cocok = [r for r in results if not r["benar"] and r["expected"] is not None]
    gagal_diluar = [r for r in results if not r["benar"] and r["expected"] is None]

    if gagal_cocok or gagal_diluar:
        print("  📋 SARAN PERBAIKAN:")
        for r in gagal_cocok:
            print(f"     → Tambah variasi '{r['pertanyaan']}' ke intent {r['expected']}")
        for r in gagal_diluar:
            print(f"     → '{r['pertanyaan']}' (skor {r['skor']:.3f}) tertangkap sebagai {r['hasil']}")
            print(f"       Tambah ke intent OUT-001 (fallback) atau perbanyak variasi intent terkait")

    print()


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluasi threshold chatbot BESTI")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Uji threshold tertentu (default: gunakan rekomendasi otomatis)")
    parser.add_argument("--verbose",   action="store_true",
                        help="Tampilkan semua hasil pengujian, bukan hanya yang gagal")
    args = parser.parse_args()

    # Load & build
    data, all_questions, all_labels = load_intents(INTENTS_FILE)
    vectorizer, question_vectors    = build_vectorizer(all_questions)

    # Tentukan threshold
    if args.threshold:
        threshold = args.threshold
    else:
        # Hitung otomatis dari data dulu (tanpa threshold)
        _, scores_cocok_awal, scores_diluar_awal = run_evaluation(
            vectorizer, question_vectors, all_labels,
            threshold=0.0, verbose=False
        )
        skor_nilai  = [s for s, _ in scores_cocok_awal]
        threshold   = round((min(skor_nilai) + max(scores_diluar_awal)) / 2, 3)

    # Jalankan evaluasi
    results, scores_cocok, scores_diluar = run_evaluation(
        vectorizer, question_vectors, all_labels,
        threshold=threshold, verbose=args.verbose
    )

    # Tampilkan laporan
    print_report(results, scores_cocok, scores_diluar,
                 threshold, data, all_questions, args.verbose)