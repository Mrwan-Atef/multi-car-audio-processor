from app.services.plate_normalizer import (
    normalize_plate_from_text,
)


TESTS = [
    (
        "راء قاف عين 7665",
        "رقع7665",
    ),

    (
        "راء واو باء 1729",
        "روب1729",
    ),

    (
        "روابه 1729",
        "روب1729",
    ),

    (
        "بيطا صوت 5745",
        "بطص5745",
    ),

    (
        "سبعة سبعة سبعة زيرو",
        None,  # letters missing
    ),

    (
        "ب م 4567",
        None,
    ),

    (
        "سمر 123",
        None,
    ),

    (
        "سمر 12345",
        None,
    ),

    (
        "دال نون ميم سبعة تسعة ستين",
        "دنم796",
    ),

    (
        "دال نون ميم سبعة تسعة ستة زيرو",
        "دنم7960",
    ),

    (
        "الف نون لام عشرة اتنين اربعة",
        "انل1024",
    ),

    (
        "حا سين ياء تلاتين واحد خمسه",
        "حسي315",
    ),
]


for text, expected in TESTS:

    result = normalize_plate_from_text(
        text
    )

    print(
        f"{text} => {result}"
    )

    assert result == expected