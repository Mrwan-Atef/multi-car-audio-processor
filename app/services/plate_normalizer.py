import re
import unicodedata


# ============================================================
# DIGITS
# ============================================================

DIGIT_WORDS = {
    "صفر": "0",
    "زيرو": "0",

    "واحد": "1",

    "اثنين": "2",
    "اثنان": "2",
    "اتنين": "2",

    "ثلاثة": "3",
    "ثلاثه": "3",
    "تلاتة": "3",
    "تلاته": "3",

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


TENS_WORDS = {
    "عشرة": "10",
    "عشره": "10",
    "عشر": "10",

    "عشرين": "20",
    "عشرون": "20",

    "ثلاثين": "30",
    "ثلاثون": "30",
    "تلاتين": "30",

    "أربعين": "40",
    "اربعين": "40",
    "أربعون": "40",
    "اربعون": "40",

    "خمسين": "50",
    "خمسون": "50",

    "ستين": "60",
    "ستون": "60",

    "سبعين": "70",
    "سبعون": "70",

    "ثمانين": "80",
    "ثمانون": "80",

    "تسعين": "90",
    "تسعون": "90",
}


ONES_VALUES = {
    "واحد": 1,
    "اثنين": 2,
    "اثنان": 2,
    "اتنين": 2,
    "ثلاثة": 3,
    "ثلاثه": 3,
    "تلاتة": 3,
    "تلاته": 3,
    "أربعة": 4,
    "اربعة": 4,
    "أربعه": 4,
    "اربعه": 4,
    "خمسة": 5,
    "خمسه": 5,
    "ستة": 6,
    "سته": 6,
    "سبعة": 7,
    "سبعه": 7,
    "ثمانية": 8,
    "ثمانيه": 8,
    "تمانية": 8,
    "تمانيه": 8,
    "تمنية": 8,
    "تسعة": 9,
    "تسعه": 9,
}


DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


# ============================================================
# LETTERS
# ============================================================

LETTER_WORDS = {
    "ا": "ا",
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "الف": "ا",
    "ألف": "ا",

    "ب": "ب",
    "باء": "ب",
    "بي": "ب",
    "به": "ب",

    "ت": "ت",
    "تاء": "ت",
    "تي": "ت",

    "ث": "ث",
    "ثاء": "ث",

    "ج": "ج",
    "جيم": "ج",

    "ح": "ح",
    "حاء": "ح",
    "حا": "ح",

    "خ": "خ",
    "خاء": "خ",

    "د": "د",
    "دال": "د",
    "دا": "د",
    "ده": "د",

    "ذ": "ذ",
    "ذال": "ذ",

    "ر": "ر",
    "راء": "ر",
    "را": "ر",

    "ز": "ز",
    "زاي": "ز",

    "س": "س",
    "سين": "س",

    "ش": "ش",
    "شين": "ش",

    "ص": "ص",
    "صاد": "ص",

    "ض": "ض",
    "ضاد": "ض",

    "ط": "ط",
    "طاء": "ط",
    "طا": "ط",
    "طي": "ط",

    "ظ": "ظ",
    "ظاء": "ظ",

    "ع": "ع",
    "عين": "ع",

    "غ": "غ",
    "غين": "غ",

    "ف": "ف",
    "فاء": "ف",

    "ق": "ق",
    "قاف": "ق",

    "ك": "ك",
    "كاف": "ك",

    "ل": "ل",
    "لام": "ل",

    "م": "م",
    "ميم": "م",

    "ن": "ن",
    "نون": "ن",

    "ه": "ه",
    "هاء": "ه",

    "و": "و",
    "واو": "و",

    "ي": "ي",
    "ياء": "ي",
}


# ============================================================
# CLEANING
# ============================================================

def _clean_token(
    token: str,
) -> str:

    token = token.strip()

    token = token.translate(
        DIGIT_TRANSLATION
    )

    token = "".join(
        char
        for char in token
        if not unicodedata.combining(char)
    )

    token = re.sub(
        r"[،,.;:!?()\[\]{}\"'ـ]",
        "",
        token,
    )

    return token.strip()


# ============================================================
# DIGITS
# ============================================================

def normalize_digit_token(
    token: str,
) -> str | None:

    token = _clean_token(token)

    if not token:
        return None

    if token in DIGIT_WORDS:
        return DIGIT_WORDS[token]

    if token in TENS_WORDS:
        return TENS_WORDS[token]

    if token.isdigit():
        return token

    return None


def is_digit_token(
    token: str,
) -> bool:

    return (
        normalize_digit_token(token)
        is not None
    )


def normalize_number_expression(
    tokens: list[str],
    start: int,
) -> tuple[str | None, int]:

    if start >= len(tokens):
        return None, start

    first = _clean_token(
        tokens[start]
    )

    # سبعة وسبعين -> 77
    if start + 2 < len(tokens):

        second = _clean_token(
            tokens[start + 1]
        )

        third = _clean_token(
            tokens[start + 2]
        )

        if (
            first in ONES_VALUES
            and second == "و"
            and third in TENS_WORDS
        ):

            return (
                str(
                    int(
                        TENS_WORDS[third]
                    )
                    + ONES_VALUES[first]
                ),
                start + 3,
            )

    value = normalize_digit_token(
        first
    )

    if value is None:
        return None, start

    return (
        value,
        start + 1,
    )


def normalize_digits_from_tokens(
    tokens: list[str],
) -> str | None:

    digits = []

    index = 0

    while index < len(tokens):

        value, next_index = (
            normalize_number_expression(
                tokens,
                index,
            )
        )

        if value is None:
            return None

        digits.extend(
            value
        )

        if len(digits) > 4:
            return None

        index = next_index

    if len(digits) != 4:
        return None

    return "".join(digits)


# ============================================================
# LETTERS
# ============================================================

def normalize_letter_token_exact(
    token: str,
) -> list[str]:

    token = _clean_token(
        token
    )

    if not token:
        return []

    value = LETTER_WORDS.get(
        token
    )

    if value:
        return [value]

    return []


def decode_merged_letters(
    tokens: list[str],
) -> list[str]:

    """
    Generic, conservative decoder.

    It is deliberately based on exact letter names first.
    It does not contain any plate-specific mappings.
    """

    letters = []

    for token in tokens:

        value = normalize_letter_token_exact(
            token
        )

        if not value:
            return []

        letters.extend(
            value
        )

    if len(letters) == 3:
        return letters

    return []


# ============================================================
# PLATE
# ============================================================

def normalize_plate_from_text(
    text: str,
) -> str | None:

    text = text.strip()

    if not text:
        return None

    tokens = text.split()

    # --------------------------------------------------------
    # Search every possible prefix.
    # --------------------------------------------------------

    for letter_count in range(
        1,
        min(
            8,
            len(tokens),
        ) + 1,
    ):

        letter_tokens = tokens[
            :letter_count
        ]

        if any(
            is_digit_token(token)
            for token in letter_tokens
        ):
            break

        if letter_count >= len(tokens):
            continue

        # Digits must immediately follow.
        if not is_digit_token(
            tokens[letter_count]
        ):
            continue

        letters = decode_merged_letters(
            letter_tokens
        )

        if len(letters) != 3:
            continue

        digits = normalize_digits_from_tokens(
            tokens[letter_count:]
        )

        if digits is None:
            continue

        return (
            "".join(letters)
            + digits
        )

    return None