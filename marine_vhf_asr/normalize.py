"""
Transcript normalization for marine VHF radio.

Follows the same conventions as the public ATC ASR datasets
(jacktol/ATC-ASR-Dataset) so the two can be mixed during training:
- Uppercase, no punctuation
- Digits spelled out one by one ("16" -> "ONE SIX")
- Isolated letters expanded to the ITU/NATO phonetic alphabet ("N" -> "NOVEMBER")

Maritime specifics:
- PAN-PAN -> PAN PAN, SÉCURITÉ / SECURITAY -> SECURITE
- NINER -> NINE (spoken variant, same meaning)
- "." between digits -> POINT, "°" -> DEGREES, "'" after digits -> MINUTES
"""

import re
import unicodedata

DIGITS = {
    "0": "ZERO", "1": "ONE", "2": "TWO", "3": "THREE", "4": "FOUR",
    "5": "FIVE", "6": "SIX", "7": "SEVEN", "8": "EIGHT", "9": "NINE",
}

PHONETIC = {
    "A": "ALFA", "B": "BRAVO", "C": "CHARLIE", "D": "DELTA", "E": "ECHO",
    "F": "FOXTROT", "G": "GOLF", "H": "HOTEL", "I": "INDIA", "J": "JULIETT",
    "K": "KILO", "L": "LIMA", "M": "MIKE", "N": "NOVEMBER", "O": "OSCAR",
    "P": "PAPA", "Q": "QUEBEC", "R": "ROMEO", "S": "SIERRA", "T": "TANGO",
    "U": "UNIFORM", "V": "VICTOR", "W": "WHISKEY", "X": "XRAY", "Y": "YANKEE",
    "Z": "ZULU",
}

# Spelling variants that should score as the same word.
CANONICAL = {
    "ALPHA": "ALFA",
    "JULIET": "JULIETT",
    "WHISKY": "WHISKEY",
    "X-RAY": "XRAY",
    "NINER": "NINE",
    "SECURITAY": "SECURITE",
    "SAYCURITAY": "SECURITE",
    "PAN-PAN": "PAN PAN",
    "MAYDAY-RELAY": "MAYDAY RELAY",
    "OK": "OKAY",
}

# Single-letter words that should not be expanded to the phonetic alphabet.
_KEEP_SINGLE = {"A", "I"}


def _strip_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


_ONES = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
         "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN",
         "SEVENTEEN", "EIGHTEEN", "NINETEEN"]
_TENS = ["", "", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"]


def cardinal(number: str) -> str:
    """
    Read a digit string as an English cardinal: '31' -> 'THIRTY ONE'.

    Numbers with a leading zero or above 999999 are spelled digit by digit,
    which is how they are read on the radio anyway.
    """
    n = int(number)
    if (len(number) > 1 and number[0] == "0") or n > 999999:
        return spell_digits(number)

    def below_1000(n):
        words = []
        if n >= 100:
            words += [_ONES[n // 100], "HUNDRED"]
            n %= 100
        if n >= 20:
            words.append(_TENS[n // 10])
            n %= 10
            if n:
                words.append(_ONES[n])
        elif n or not words:
            words.append(_ONES[n])
        return words

    if n >= 1000:
        thousands, rest = divmod(n, 1000)
        words = below_1000(thousands) + ["THOUSAND"] + (below_1000(rest) if rest else [])
    else:
        words = below_1000(n)
    return " ".join(words)


def normalize_transcript(
    text: str,
    expand_letters: bool = True,
    numbers: str = "digits",
    acronyms: bool = False,
) -> str:
    """
    Normalize a transcript to the canonical training/eval form.

    Args:
        text: Raw transcript
        expand_letters: Expand isolated letters (other than A/I) to phonetic words
        numbers: "digits" spells numbers digit by digit ("16" -> "ONE SIX", the
            radio convention); "cardinal" reads them as words ("16" -> "SIXTEEN"),
            for sources such as TTS corpora where that is what was spoken.
            Digits after a decimal point are always spelled one by one.
        acronyms: Join runs of spaced single letters into one word
            ("M M S I" -> "MMSI") instead of expanding them phonetically

    Returns:
        Uppercase, punctuation-free transcript with numbers spelled out
    """
    if numbers not in ("digits", "cardinal"):
        raise ValueError(f"numbers must be 'digits' or 'cardinal', got {numbers!r}")
    t = _strip_diacritics(text).upper()

    if acronyms:
        t = re.sub(r"(?<![A-Z'])[A-Z](?![A-Z'])(?:[\s-]+[A-Z](?![A-Z']))+",
                   lambda m: re.sub(r"[\s-]", "", m.group()), t)

    # Hyphenated canonical forms must be handled before hyphens are stripped
    for variant in ("PAN-PAN", "MAYDAY-RELAY", "X-RAY"):
        t = t.replace(variant, CANONICAL[variant])

    # Coordinates and decimals: 50°12.5'N
    hemispheres = {"N": "NORTH", "S": "SOUTH", "E": "EAST", "W": "WEST"}
    t = re.sub(r"(?<=[°'′])\s*([NSEW])\b", lambda m: f" {hemispheres[m.group(1)]} ", t)
    t = re.sub(r"(?<=\d)\s*°", " DEGREES ", t)
    t = re.sub(r"(?<=\d)\s*['′]", " MINUTES ", t)
    t = re.sub(r"(?<=\d)\.(?=\d)", " POINT ", t)

    # Split digits off from letters ("CH16" -> "CH 16") then spell the numbers
    t = re.sub(r"(?<=[A-Z])(?=\d)|(?<=\d)(?=[A-Z])", " ", t)
    t = re.sub(r"(?<=POINT )\d+", lambda m: f" {spell_digits(m.group())} ", t)
    if numbers == "cardinal":
        t = re.sub(r"\d+", lambda m: f" {cardinal(m.group())} ", t)
    t = re.sub(r"\d", lambda m: f" {DIGITS[m.group()]} ", t)

    # Remove punctuation (keep apostrophes inside words, e.g. "WE'RE")
    t = re.sub(r"[^A-Z' ]+", " ", t)
    t = re.sub(r"(?<![A-Z])'|'(?![A-Z])", " ", t)

    words = []
    for w in t.split():
        w = CANONICAL.get(w, w)
        if expand_letters and len(w) == 1 and w not in _KEEP_SINGLE:
            w = PHONETIC.get(w, w)
        words.extend(w.split())
    return " ".join(words)


def spell_digits(number: str) -> str:
    """Spell a digit string one digit at a time: '235' -> 'TWO THREE FIVE'."""
    return " ".join(DIGITS[d] for d in number if d in DIGITS)


def spell_letters(letters: str) -> str:
    """Spell letters/digits phonetically: 'PD2' -> 'PAPA DELTA TWO'."""
    out = []
    for c in letters.upper():
        if c in PHONETIC:
            out.append(PHONETIC[c])
        elif c in DIGITS:
            out.append(DIGITS[c])
    return " ".join(out)
