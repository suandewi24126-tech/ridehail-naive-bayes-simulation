import pandas as pd

from ridehail_nb.modeling import fit_and_evaluate, predict_review


def test_model_can_fit_and_predict():
    rows = [
        ("Aplikasi mudah digunakan dan proses cepat", 0),
        ("Driver tiba sesuai lokasi penjemputan", 0),
        ("Pembayaran berhasil tercatat di aplikasi", 0),
        ("Fitur lokasi menunjukkan alamat tujuan", 0),
        ("Saya sangat kecewa, aplikasinya buruk", 1),
        ("Pelayanan mengecewakan dan membuat kesal", 1),
        ("Driver ramah sekali, saya sangat senang", 1),
        ("Aplikasi ini menyebalkan dan lambat sekali", 1),
        ("Riwayat perjalanan tersimpan otomatis", 0),
        ("Tarif perjalanan muncul sebelum pemesanan", 0),
        ("Saya puas dan suka memakai layanan ini", 1),
        ("Pengalaman saya sangat buruk hari ini", 1),
    ]
    df = pd.DataFrame(rows, columns=["review", "label"])
    result = fit_and_evaluate(df, alpha=1.0, test_size=0.25, random_state=42)
    prediction = predict_review(result["model"], "Saya sangat kecewa dengan layanan ini")
    assert prediction["label"] in (0, 1)
    assert 0.0 <= prediction["confidence"] <= 1.0
