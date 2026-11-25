from ridehail_nb.preprocessing import clean_text


def test_clean_text_normalizes_common_abbreviations_and_keeps_negation():
    result = clean_text("Gk bagus bgt! Cek https://contoh.id")
    assert result == "tidak bagus banget cek"
