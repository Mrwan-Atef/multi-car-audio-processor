import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

# ============================================================
# CLEANING
# ============================================================

DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


def _clean_token(token: str) -> str:
    token = str(token).strip().translate(DIGIT_TRANSLATION)
    token = "".join(ch for ch in token if not unicodedata.combining(ch))
    token = re.sub(r"<[^>]+>", "", token)
    token = re.sub(r"[،,.;:!?()\[\]{}\"'ـ]", "", token)
    return token.strip()


def clean_transcript_text(text: str) -> str:
    text = str(text).translate(DIGIT_TRANSLATION)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[،,.;:!?()\[\]{}\"'ـ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _phonetic(text: str) -> str:
    return (
        _clean_token(text)
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )


# ============================================================
# DIGITS
# ============================================================

DIGIT_WORDS = {
    "صفر": "0", "زيرو": "0",
    "واحد": "1", "واحده": "1",
    "اثنين": "2", "اثنان": "2", "اتنين": "2",
    "ثلاثة": "3", "ثلاثه": "3", "تلاتة": "3", "تلاته": "3", "تلات": "3",
    "أربعة": "4", "اربعة": "4", "أربعه": "4", "اربعه": "4", "اربع": "4",
    "خمسة": "5", "خمسه": "5", "خمس": "5",
    "ستة": "6", "سته": "6", "ست": "6",
    "سبعة": "7", "سبعه": "7", "سبع": "7",
    "ثمانية": "8", "ثمانيه": "8", "تمانية": "8", "تمانيه": "8", "تمنية": "8", "تمان": "8",
    "تسعة": "9", "تسعه": "9", "تسع": "9",
}

TENS_WORDS = {
    "عشرة": "10", "عشره": "10", "عشر": "10",
    "عشرين": "20", "عشرون": "20",
    "ثلاثين": "30", "ثلاثون": "30", "تلاتين": "30", "تلاتون": "30",
    "أربعين": "40", "اربعين": "40", "أربعون": "40", "اربعون": "40",
    "خمسين": "50", "خمسون": "50",
    "ستين": "60", "ستون": "60",
    "سبعين": "70", "سبعون": "70",
    "ثمانين": "80", "ثمانون": "80",
    "تسعين": "90", "تسعون": "90",
}

COMPOUND_NUMBER_WORDS = {
    "حداشر": "11",
    "أحداشر": "11",
    "احداشر": "11",
    "اتناشر": "12",
    "اتناعشر": "12",
    "تلتاشر": "13",
    "تلاتاشر": "13",
    "اربعتاشر": "14",
    "خمستاشر": "15",
    "ستاشر": "16",
    "سبعتاشر": "17",
    "تمنتاشر": "18",
    "تسعتاشر": "19",
}

ONES_VALUES = {k: int(v) for k, v in DIGIT_WORDS.items() if v != "0"}


def normalize_digit_token(token: str) -> str | None:
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


def is_digit_token(token: str) -> bool:
    return normalize_digit_token(token) is not None


def normalize_number_expression(
    tokens: list[str],
    start: int,
) -> tuple[str | None, int]:

    if start >= len(tokens):
        return None, start

    first = _clean_token(
        tokens[start]
    )

    # --------------------------------------------------------
    # Compound numbers:
    #
    # حداشر -> 11
    # اتناشر -> 12
    # ...
    # --------------------------------------------------------

    if first in COMPOUND_NUMBER_WORDS:
        return (
            COMPOUND_NUMBER_WORDS[first],
            start + 1,
        )

    # --------------------------------------------------------
    # one + و + tens
    #
    # سبعة وسبعين -> 77
    # أربعة وستين -> 64
    # --------------------------------------------------------

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

            value = (
                int(TENS_WORDS[third])
                + ONES_VALUES[first]
            )

            return (
                str(value),
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


def normalize_digits_from_tokens(tokens: list[str]) -> str | None:
    result: list[str] = []
    index = 0

    while index < len(tokens):
        value, next_index = normalize_number_expression(tokens, index)
        if value is None:
            return None
        result.extend(value)
        if len(result) > 4:
            return None
        index = next_index

    return "".join(result) if len(result) == 4 else None


# ============================================================
# LETTERS
# ============================================================

LETTER_WORDS = {
    "ا": "ا", "أ": "ا", "إ": "ا", "آ": "ا", "الف": "ا", "ألف": "ا", "الالف": "ا",
    "ب": "ب", "باء": "ب", "بي": "ب", "به": "ب", "بيه": "ب", "بهي": "ب", "با": "ب",
    "ت": "ت", "تاء": "ت", "تي": "ت", "تا": "ت",
    "ث": "ث", "ثاء": "ث", "ثا": "ث",
    "ج": "ج", "جيم": "ج",
    "ح": "ح", "حاء": "ح", "حا": "ح",
    "خ": "خ", "خاء": "خ", "خا": "خ",
    "د": "د", "دال": "د", "دا": "د", "ده": "د",
    "ذ": "ذ", "ذال": "ذ", "ذا": "ذ",
    "ر": "ر", "راء": "ر", "را": "ر",
    "ز": "ز", "زاي": "ز", "زا": "ز",
    "س": "س", "سين": "س", "سي": "س", "سا": "س",
    "ش": "ش", "شين": "ش", "شي": "ش", "شا": "ش",
    "ص": "ص", "صاد": "ص", "صوت": "ص", "صا": "ص",
    "ض": "ض", "ضاد": "ض", "ضا": "ض",
    "ط": "ط", "طاء": "ط", "طا": "ط", "طي": "ط",
    "ظ": "ظ", "ظاء": "ظ", "ظا": "ظ",
    "ع": "ع", "عين": "ع", "عا": "ع",
    "غ": "غ", "غين": "غ", "غا": "غ",
    "ف": "ف", "فاء": "ف", "فا": "ف",
    "ق": "ق", "قاف": "ق", "قا": "ق",
    "ك": "ك", "كاف": "ك", "كيف": "ك", "كا": "ك",
    "ل": "ل", "لام": "ل", "لا": "ل",
    "م": "م", "ميم": "م", "مي": "م", "ما": "م",
    "ن": "ن", "نون": "ن", "نو": "ن", "نا": "ن",
    "ه": "ه", "هاء": "ه", "ها": "ه", "اله": "ه",
    "و": "و", "واو": "و", "الواو": "و", "وا": "و",
    "ي": "ي", "ياء": "ي", "يا": "ي",
}

# Short forms allowed while decoding a merged ASR token.
# Long/ambiguous forms such as "بهي" are deliberately absent here.
COMPOSE_FORMS = {
    "ا": ["ا"], "ب": ["ب"], "ت": ["ت"], "ث": ["ث"], "ج": ["ج"],
    "ح": ["ح", "حا"], "خ": ["خ", "خا"], "د": ["د"], "ذ": ["ذ"],
    "ر": ["ر", "را"], "ز": ["ز", "زا"], "س": ["س", "سا"], "ش": ["ش", "شا"],
    "ص": ["ص", "صا"], "ض": ["ض", "ضا"], "ط": ["ط", "طا"], "ظ": ["ظ", "ظا"],
    "ع": ["ع", "عا"], "غ": ["غ", "غا"], "ف": ["ف", "فا"], "ق": ["ق", "قا"],
    "ك": ["ك", "كا"], "ل": ["ل", "لا"], "م": ["م", "ما"], "ن": ["ن", "نا"],
    "ه": ["ه", "ها"], "و": ["و", "وا"], "ي": ["ي", "يا"],
}



_CARRIER_PREFIXES = {
    "حا": "ح", "خا": "خ", "را": "ر", "زا": "ز", "سا": "س",
    "شا": "ش", "صا": "ص", "ضا": "ض", "طا": "ط", "ظا": "ظ",
    "عا": "ع", "غا": "غ", "فا": "ف", "قا": "ق", "كا": "ك",
    "لا": "ل", "ما": "م", "نا": "ن", "ها": "ه", "وا": "و", "يا": "ي",
}

@dataclass(frozen=True)
class LetterOption:
    letters: tuple[str, ...]
    cost: int
    strong: bool
    kind: str


@dataclass(frozen=True)
class PhraseDecode:
    letters: tuple[str, ...]
    cost: int
    repeated: bool
    strong: bool


def _literal_letters(text: str) -> tuple[str, ...] | None:
    """Literal Arabic characters only. No semantic guessing."""
    text = _phonetic(text)
    if 1 <= len(text) <= 3 and all(ch in LETTER_WORDS for ch in text):
        return tuple(LETTER_WORDS[ch] for ch in text)
    return None


def _compose_nonambiguous(text: str, max_letters: int = 3) -> list[tuple[str, ...]]:
    """Compose using single Arabic characters and short carrier forms."""
    text = _phonetic(text)
    forms: list[tuple[str, str]] = []
    for letter, variants in COMPOSE_FORMS.items():
        for variant in variants:
            forms.append((_phonetic(variant), letter))

    result: list[tuple[str, ...]] = []

    def walk(pos: int, seq: tuple[str, ...]) -> None:
        if pos == len(text):
            if 1 <= len(seq) <= max_letters:
                result.append(seq)
            return
        if len(seq) >= max_letters:
            return
        for form, letter in forms:
            if text.startswith(form, pos):
                walk(pos + len(form), seq + (letter,))

    walk(0, ())
    return list(dict.fromkeys(result))


def _add_option(store: dict[tuple[str, ...], LetterOption], seq: tuple[str, ...], cost: int, strong: bool) -> None:
    old = store.get(seq)
    new = LetterOption(seq, cost, strong)
    if old is None or (new.cost, not new.strong) < (old.cost, not old.strong):
        store[seq] = new


@lru_cache(maxsize=8192)
def _token_letter_options(token: str) -> tuple[LetterOption, ...]:
    raw = _clean_token(token)
    text = _phonetic(raw)
    if not text:
        return ()

    options: dict[tuple[str, ...], LetterOption] = {}

    def add(seq: tuple[str, ...], cost: int, strong: bool, kind: str) -> None:
        if not (1 <= len(seq) <= 3):
            return
        old = options.get(seq)
        new = LetterOption(seq, cost, strong, kind)
        if old is None or (new.cost, not new.strong) < (old.cost, not old.strong):
            options[seq] = new

    exact = LETTER_WORDS.get(raw)
    if exact:
        add((exact,), 0, True, "exact")

    # Generic carrier merge.
    if len(text) >= 3 and text[-1] in {"ي", "ا", "و"}:
        reduced = text[:-1]
        literal = _literal_letters(reduced)
        if literal and len(literal) == 2:
            add(literal, 1 if exact else 0, False, "carrier_merge")

    # Generic one-character deletion.
    n = len(text)
    for i in range(n):
        reduced = text[:i] + text[i + 1:]
        base_cost = 0 if i == n - 1 else (1 if i > 0 else 2)

        exact_reduced = LETTER_WORDS.get(reduced)
        if exact_reduced:
            add((exact_reduced,), base_cost + (2 if exact else 0), False, "deletion_exact")
        else:
            literal = _literal_letters(reduced)
            if literal:
                if len(text) == 3 and len(literal) == 2:
                    add(literal, base_cost + (2 if exact else 0), False, "deletion_literal")
                elif len(text) >= 4 and len(literal) == 3:
                    add(literal, base_cost + (2 if exact else 0), False, "deletion_literal")

        # Carrier-prefix + one final literal letter. This is useful for
        # generic ASR merges such as حاسن -> حا + س. It is marked specially so
        # the repeated-plate rule can reject it when repetition contradicts it.
        if len(text) >= 4:
            for prefix, first_letter in _CARRIER_PREFIXES.items():
                if reduced.startswith(prefix) and len(reduced) == len(prefix) + 1:
                    last = reduced[-1]
                    if last in LETTER_WORDS:
                        add(
                            (first_letter, LETTER_WORDS[last]),
                            base_cost + (2 if exact else 0),
                            False,
                            "carrier_prefix",
                        )

    # Two-character deletion. Prefer deleting a suffix; this is particularly
    # useful when the whole surrounding phrase proves 3+3 repetition.
    if n >= 4:
        for i in range(n):
            for j in range(i + 1, n):
                reduced = "".join(ch for k, ch in enumerate(text) if k not in {i, j})
                def del_cost(pos: int) -> int:
                    return 1 if pos >= n - 2 else (2 if pos > 0 else 3)
                cost = del_cost(i) + del_cost(j)
                exact_reduced = LETTER_WORDS.get(reduced)
                if exact_reduced:
                    add((exact_reduced,), cost + (2 if exact else 0), False, "double_deletion")
                    # A two-character exact token such as را can also be the
                    # literal pair ر + ا when the surrounding phrase proves
                    # that more than one letter is being spoken.
                    if len(reduced) == 2:
                        literal = _literal_letters(reduced)
                        if literal:
                            add(literal, cost + (2 if exact else 0), False, "double_deletion")
                else:
                    literal = _literal_letters(reduced)
                    if literal and len(literal) == 2:
                        add(literal, cost + (2 if exact else 0), False, "double_deletion")

    # Weak direct 3-character token fallback.
    if len(text) == 3:
        literal = _literal_letters(text)
        if literal:
            add(literal, 3 + (2 if exact else 0), False, "literal_fallback")

    return tuple(sorted(options.values(), key=lambda x: (x.cost, not x.strong, len(x.letters))))

def decode_token(token: str, max_letters: int = 3) -> list[list[str]]:
    return [list(o.letters) for o in _token_letter_options(token) if len(o.letters) <= max_letters]


def decode_letter_phrase_scored(tokens: list[str]) -> PhraseDecode | None:
    if not tokens or len(tokens) > 8:
        return None

    option_lists = [_token_letter_options(t) for t in tokens]
    if any(not opts for opts in option_lists):
        return None

    best: tuple[tuple, PhraseDecode] | None = None

    def consider(seq: tuple[str, ...], cost: int, strong: bool, kinds: tuple[str, ...]) -> None:
        nonlocal best

        if len(seq) == 3:
            candidate = PhraseDecode(seq, cost, False, strong)
        elif len(seq) == 6 and seq[:3] == seq[3:]:
            # A repeated plate must not depend on the ambiguous carrier-prefix
            # repair. Otherwise a normal Arabic token can become a fake 3+3
            # plate merely because repetition happens to fit.
            if "carrier_prefix" in kinds:
                return
            candidate = PhraseDecode(seq[:3], cost, True, True)
        else:
            return

        rank = (
            candidate.cost - (1 if candidate.repeated else 0),
            -int(candidate.strong),
            0 if candidate.repeated else 1,
        )
        if best is None or rank < best[0]:
            best = (rank, candidate)

    def walk(index: int, seq: tuple[str, ...], cost: int, strong: bool, kinds: tuple[str, ...]) -> None:
        if len(seq) > 6 or cost > 8:
            return
        if index == len(option_lists):
            consider(seq, cost, strong, kinds)
            return
        for option in option_lists[index]:
            next_seq = seq + option.letters
            if len(next_seq) <= 6:
                walk(
                    index + 1,
                    next_seq,
                    cost + option.cost,
                    strong and option.strong,
                    kinds + (option.kind,),
                )

    walk(0, (), 0, True, ())
    return best[1] if best else None

def decode_letter_phrase(tokens: list[str]) -> list[str]:
    decoded = decode_letter_phrase_scored(tokens)
    return list(decoded.letters) if decoded else []


def decode_merged_letters(tokens: list[str]) -> list[str]:
    return decode_letter_phrase(tokens)


def normalize_plate_from_text(text: str) -> str | None:
    tokens = clean_transcript_text(text).split()
    if len(tokens) < 2:
        return None

    # Candidate text is expected to end in exactly four spoken digits.
    for digit_start in range(1, len(tokens)):
        digits = normalize_digits_from_tokens(tokens[digit_start:])
        if digits is None:
            continue

        decoded = decode_letter_phrase_scored(tokens[:digit_start])
        if decoded is None:
            continue

        # Reject weak one-word Arabic text being mistaken for three letters.
        if len(tokens[:digit_start]) == 1 and not decoded.strong:
            continue

        return "".join(decoded.letters) + digits

    return None
