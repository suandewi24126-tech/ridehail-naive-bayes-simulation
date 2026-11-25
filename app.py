from __future__ import annotations

from pathlib import Path
import hashlib
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ridehail_nb.data import DatasetValidationError, class_distribution, load_reviews
from ridehail_nb.modeling import fit_and_evaluate, predict_review
from ridehail_nb.simulation import run_simulations, summarize_simulations

DEMO_PATH = ROOT / "data" / "demo_reviews.csv"

st.set_page_config(
    page_title="Ride-Hailing Review Lab",
    page_icon="🛵",
    layout="wide",
)

st.title("Klasifikasi Subjektivitas Ulasan Ride-Hailing")
st.caption("Pemodelan dan simulasi menggunakan TF-IDF + Multinomial Naive Bayes")

with st.sidebar:
    st.header("Sumber data")
    uploaded = st.file_uploader("Unggah dataset CSV", type=["csv"])
    st.caption(
        "CSV minimal memiliki kolom `review` dan `label`. "
        "Label: 0=Objektif, 1=Subjektif."
    )
    page = st.radio(
        "Navigasi",
        ["Ringkasan data", "Pelatihan & prediksi", "Simulasi eksperimen"],
    )

try:
    if uploaded is not None:
        df = load_reviews(uploaded)
        data_source = f"Upload: {uploaded.name}"
        data_key = f"upload:{uploaded.name}:{hashlib.sha256(uploaded.getvalue()).hexdigest()}"
    else:
        df = load_reviews(DEMO_PATH)
        data_source = "Dataset demo sintetis"
        data_key = "demo"
except Exception as exc:
    st.error(f"Dataset tidak dapat dibaca: {exc}")
    st.stop()

if st.session_state.get("data_key") != data_key:
    st.session_state["data_key"] = data_key
    st.session_state.pop("trained_model", None)
    st.session_state.pop("train_result", None)
    st.session_state.pop("simulation_raw", None)

if page == "Ringkasan data":
    st.subheader("Ringkasan dataset")
    st.info(
        f"Sumber aktif: **{data_source}**. Dataset demo hanya untuk mencoba aplikasi; "
        "jangan gunakan hasilnya sebagai hasil penelitian."
        if uploaded is None
        else f"Sumber aktif: **{data_source}**. Pastikan dataset memiliki izin penggunaan yang sesuai."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah ulasan", f"{len(df):,}")
    c2.metric("Ulasan objektif", f"{int((df['label'] == 0).sum()):,}")
    c3.metric("Ulasan subjektif", f"{int((df['label'] == 1).sum()):,}")

    st.markdown("#### Distribusi kelas")
    dist = class_distribution(df)
    st.dataframe(dist, use_container_width=True, hide_index=True)
    st.bar_chart(dist.set_index("kelas")["jumlah"])

    st.markdown("#### Pratinjau data")
    show_cols = [c for c in ["app", "score", "review", "label"] if c in df.columns]
    preview = df[show_cols].copy()
    preview["kelas"] = preview["label"].map({0: "Objektif", 1: "Subjektif"})
    st.dataframe(preview.head(20), use_container_width=True, hide_index=True)

    with st.expander("Format dataset yang diharapkan"):
        st.code(
            "score,app,review,label\n"
            "5,Gojek,Aplikasinya mudah digunakan,0\n"
            "1,Grab,Saya sangat kecewa dengan pelayanannya,1",
            language="csv",
        )
        st.write(
            "Kolom `score` dan `app` bersifat tambahan. Kolom wajib adalah `review` "
            "dan `label`. Konvensi label mengikuti laporan kelompok sebelumnya: "
            "0 = objektif dan 1 = subjektif."
        )

elif page == "Pelatihan & prediksi":
    st.subheader("Latih dan evaluasi model")
    left, right = st.columns([1, 1])
    with left:
        alpha = st.select_slider("Parameter alpha", options=[0.1, 0.5, 1.0, 2.0, 5.0], value=1.0)
    with right:
        test_size = st.select_slider("Proporsi data uji", options=[0.2, 0.25, 0.3, 0.4], value=0.2)

    if st.button("Latih model dan evaluasi", type="primary"):
        with st.spinner("Melatih dan mengevaluasi model..."):
            try:
                result = fit_and_evaluate(df, alpha=float(alpha), test_size=float(test_size))
                st.session_state["trained_model"] = result["model"]
                st.session_state["train_result"] = result
                st.success("Model berhasil dilatih.")
            except Exception as exc:
                st.error(f"Pelatihan gagal: {exc}")

    result = st.session_state.get("train_result")
    if result is not None:
        st.markdown("#### Hasil evaluasi")
        m = result["metrics"]
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{m['accuracy'] * 100:.2f}%")
        cols[1].metric("Macro Precision", f"{m['macro_precision'] * 100:.2f}%")
        cols[2].metric("Macro Recall", f"{m['macro_recall'] * 100:.2f}%")
        cols[3].metric("Macro F1", f"{m['macro_f1'] * 100:.2f}%")
        st.write(
            f"Data latih: **{result['n_train']}** | Data uji: **{result['n_test']}** | "
            f"Waktu training: **{result['training_seconds']:.4f} detik** | "
            f"Waktu prediksi: **{result['prediction_seconds']:.4f} detik**"
        )

        cm = pd.DataFrame(
            result["confusion_matrix"],
            index=["Aktual Objektif", "Aktual Subjektif"],
            columns=["Prediksi Objektif", "Prediksi Subjektif"],
        )
        st.markdown("#### Confusion matrix")
        st.dataframe(cm, use_container_width=True)

        report = pd.DataFrame(result["classification_report"]).T
        st.markdown("#### Classification report")
        st.dataframe(report.round(4), use_container_width=True)

        st.download_button(
            "Unduh hasil evaluasi (CSV)",
            data=report.to_csv().encode("utf-8-sig"),
            file_name="classification_report_naive_bayes.csv",
            mime="text/csv",
        )

    st.divider()
    st.markdown("#### Prediksi ulasan baru")
    model = st.session_state.get("trained_model")
    review_text = st.text_area(
        "Masukkan ulasan ride-hailing",
        placeholder="Contoh: Driver datang cepat dan sangat ramah.",
        height=110,
    )
    if st.button("Prediksi ulasan"):
        if model is None:
            st.warning("Latih model terlebih dahulu pada bagian di atas.")
        else:
            try:
                pred = predict_review(model, review_text)
                st.success(f"Hasil klasifikasi: **{pred['label_name']}**")
                st.write(f"Confidence model: **{pred['confidence'] * 100:.2f}%**")
                st.write(
                    f"Probabilitas objektif: {pred['probability_objective'] * 100:.2f}%  |  "
                    f"Probabilitas subjektif: {pred['probability_subjective'] * 100:.2f}%"
                )
            except ValueError as exc:
                st.warning(str(exc))

elif page == "Simulasi eksperimen":
    st.subheader("Simulasi pengaruh parameter dan kondisi data")
    st.write(
        "Simulasi membandingkan satu faktor pada satu waktu. Untuk setiap seed, "
        "data uji 20% dipertahankan saat variasi alpha, ukuran data latih, dan "
        "kondisi distribusi kelas dibandingkan. Undersampling hanya diterapkan "
        "pada data latih."
    )
    seeds_count = st.selectbox("Jumlah pengulangan (random seed)", [3, 5], index=1)
    if st.button("Jalankan seluruh simulasi", type="primary"):
        with st.spinner("Menjalankan eksperimen; beberapa model akan dilatih..."):
            try:
                seeds = tuple(range(42, 42 + int(seeds_count)))
                raw = run_simulations(df, seeds=seeds)
                summary = summarize_simulations(raw)
                st.session_state["simulation_raw"] = raw
                st.session_state["simulation_summary"] = summary
                st.success(f"Simulasi selesai: {len(raw)} proses pelatihan.")
            except Exception as exc:
                st.error(f"Simulasi gagal: {exc}")

    summary = st.session_state.get("simulation_summary")
    raw = st.session_state.get("simulation_raw")
    if summary is not None and raw is not None:
        st.markdown("#### Ringkasan rata-rata hasil simulasi")
        st.dataframe(summary.round(4), use_container_width=True, hide_index=True)

        alpha_summary = summary[summary["experiment"] == "Variasi alpha"].copy()
        if not alpha_summary.empty:
            st.markdown("#### Pengaruh alpha terhadap Macro F1")
            alpha_summary["parameter"] = alpha_summary["parameter"].astype(float)
            chart = alpha_summary.sort_values("parameter").set_index("parameter")[["macro_f1_mean"]]
            st.line_chart(chart)

        size_summary = summary[summary["experiment"] == "Variasi jumlah data latih"].copy()
        if not size_summary.empty:
            st.markdown("#### Pengaruh jumlah data latih terhadap Macro F1")
            size_summary["parameter"] = size_summary["parameter"].astype(float)
            chart = size_summary.sort_values("parameter").set_index("parameter")[["macro_f1_mean"]]
            st.line_chart(chart)

        balance_summary = summary[summary["experiment"] == "Distribusi kelas training"]
        if not balance_summary.empty:
            st.markdown("#### Pengaruh undersampling pada data latih")
            st.bar_chart(balance_summary.set_index("balance")[["macro_f1_mean"]])

        st.download_button(
            "Unduh seluruh hasil simulasi (CSV)",
            data=raw.to_csv(index=False).encode("utf-8-sig"),
            file_name="hasil_simulasi_naive_bayes.csv",
            mime="text/csv",
        )
        st.download_button(
            "Unduh ringkasan simulasi (CSV)",
            data=summary.to_csv(index=False).encode("utf-8-sig"),
            file_name="ringkasan_simulasi_naive_bayes.csv",
            mime="text/csv",
        )
    else:
        st.info("Klik tombol di atas untuk menjalankan simulasi.")

st.divider()
st.caption(
    "Catatan metodologis: dataset demo bersifat sintetis dan hanya untuk uji fungsi. "
    "Gunakan dataset berlabel asli untuk hasil penelitian dan laporan UAS."
)
