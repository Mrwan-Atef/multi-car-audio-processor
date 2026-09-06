import re


SPOKEN_DIGITS = {
    "صفر": "0",
    "واحد": "1",

    "اثنين": "2",
    "اثنان": "2",
    "اتنين": "2",

    "ثلاثة": "3",
    "ثلاثه": "3",
    "تلاتة": "3",

    "أربعة": "4",
    "اربعة": "4",
    "أربعه": "4",
    "اربعه": "4",

    "خمسة": "5",
    "خمسه": "5",

    "ستة": "6",
    "سته": "6",

    "سبعة": "7",
    "سبعه": "7",

    "ثمانية": "8",
    "ثمانيه": "8",
    "تمانية": "8",
    "تمانيه": "8",
    "تمنية": "8",

    "تسعة": "9",
    "تسعه": "9",
}


def normalize_spoken_digits(text: str) -> str:
    words = text.split()

    result = []

    for word in words:
        cleaned = re.sub(
            r"[،,.!?؛:]",
            "",
            word,
        )

        result.append(
            SPOKEN_DIGITS.get(
                cleaned,
                word,
            )
        )

    return " ".join(result)