"""Turn ledger values into text a TTS engine reads correctly.

TTS engines guess at "Rs 1,24,500.50", "05/11/2026" or "INV-2041": some read the
commas as pauses, some read a date as a fraction, some say "inv". In a collections
call a misread amount is not a cosmetic bug, it is a wrong number told to a
customer. So every value the agent speaks is normalised here first, in code,
and the LLM is told to repeat the spoken form the tools return.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_DIGITS = _ONES[:10]

_ORDINALS = {1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth", 9: "ninth", 12: "twelfth"}
_MONTHS = ("January February March April May June July August September October "
           "November December").split()

CURRENCIES = {
    # code: (major unit, minor unit, uses Indian lakh/crore grouping)
    "INR": ("rupees", "paise", True),
    "USD": ("dollars", "cents", False),
    "SGD": ("Singapore dollars", "cents", False),
    "IDR": ("rupiah", "sen", False),
}


def _below_100(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + ("-" + _ONES[ones] if ones else "")


def _below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    parts = []
    if hundreds:
        parts.append(_ONES[hundreds] + " hundred")
    if rest:
        parts.append(_below_100(rest))
    return " ".join(parts)


def number_to_words(n: int, *, indian: bool = False) -> str:
    """Spell a non-negative integer. Indian grouping uses lakh and crore."""
    if n < 0:
        raise ValueError("negative amounts are not spoken")
    if n == 0:
        return "zero"
    if indian:
        scales = [(10_000_000, "crore"), (100_000, "lakh"), (1000, "thousand")]
    else:
        scales = [(1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")]
    parts = []
    for size, name in scales:
        if n >= size:
            count, n = divmod(n, size)
            parts.append(f"{number_to_words(count, indian=indian)} {name}")
    if n:
        parts.append(_below_1000(n))
    return " ".join(parts)


def speak_amount(amount: Decimal | float | str, currency: str = "INR") -> str:
    """'124500.5' INR -> 'one lakh twenty-four thousand five hundred rupees and fifty paise'."""
    major_name, minor_name, indian = CURRENCIES.get(currency, (currency, "cents", False))
    value = Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    major = int(value)
    minor = int((value - major) * 100)
    text = f"{number_to_words(major, indian=indian)} {major_name}"
    if minor:
        text += f" and {number_to_words(minor)} {minor_name}"
    return text


def _ordinal(day: int) -> str:
    if day in _ORDINALS:
        return _ORDINALS[day]
    if day < 20:
        return _ONES[day] + "th"
    tens, ones = divmod(day, 10)
    if ones == 0:
        return _TENS[tens][:-1] + "ieth"
    return _TENS[tens] + "-" + _ordinal(ones)


def speak_date(d: date, *, today: date | None = None) -> str:
    """date(2026, 11, 5) -> 'the fifth of November' (year only when it differs)."""
    text = f"the {_ordinal(d.day)} of {_MONTHS[d.month - 1]}"
    if today is None or d.year != today.year:
        if 2000 <= d.year < 2010:
            text += f", {number_to_words(d.year)}"
        else:
            text += f", {number_to_words(d.year // 100)} {_year_tail(d.year % 100)}"
    return text


def _year_tail(n: int) -> str:
    if n == 0:
        return "hundred"
    if n < 10:
        return "oh " + _ONES[n]
    return _below_100(n)


def speak_reference(ref: str) -> str:
    """'INV-2041' -> 'I N V, two zero four one'. Letters spelled, digits read singly."""
    letters = " ".join(ch for ch in ref if ch.isalpha()).upper()
    digits = " ".join(_DIGITS[int(ch)] for ch in ref if ch.isdigit())
    return ", ".join(p for p in (letters, digits) if p)


# --- streaming normaliser for anything the LLM writes itself -------------------

_MONEY_RE = re.compile(r"(?:₹|Rs\.?|INR|\$|SGD|USD)\s?([\d,]+(?:\.\d{1,2})?)")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_REF_RE = re.compile(r"\b([A-Z]{2,5})-(\d{2,8})\b")
_BARE_NUM_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_SYMBOLS = {"%": " percent", "&": " and ", "@": " at ", "#": " number "}
_MARKDOWN_RE = re.compile(r"[*_`]+")
_SENTENCE_END = re.compile(r"([.!?])\s")


def _money_sub(m: re.Match, currency: str) -> str:
    raw = m.group(0)
    cur = "USD" if raw.startswith("$") or raw.startswith("USD") else "SGD" if raw.startswith("SGD") else currency
    return speak_amount(m.group(1).replace(",", ""), cur)


def _speak_bare(token: str) -> str:
    whole, _, frac = token.replace(",", "").partition(".")
    text = number_to_words(int(whole))
    if frac:
        text += " point " + " ".join(_DIGITS[int(c)] for c in frac)
    return text


_JSON_RE = re.compile(r"\{[^{}]*\}")
_DASHES_RE = re.compile(r"[\u2010-\u2014]")


def normalize_for_tts(text: str, *, currency: str = "INR", today: date | None = None) -> str:
    """Safety net for LLM-written text: JSON, money, ISO dates, invoice refs, symbols, markdown."""
    text = _JSON_RE.sub("", text)          # a model writing tool args as text must never be read out
    text = _DASHES_RE.sub("-", text)       # 'INV‑2041' with a non-breaking hyphen is still a reference
    text = _MARKDOWN_RE.sub("", text)
    text = _MONEY_RE.sub(lambda m: _money_sub(m, currency), text)
    text = _ISO_DATE_RE.sub(
        lambda m: speak_date(date(int(m[1]), int(m[2]), int(m[3])), today=today), text)
    text = _REF_RE.sub(lambda m: speak_reference(m.group(0)), text)
    text = _BARE_NUM_RE.sub(lambda m: _speak_bare(m.group(0)), text)
    for sym, word in _SYMBOLS.items():
        text = text.replace(sym, word)
    return re.sub(r"\s{2,}", " ", text)


def unspeakable_tokens(text: str) -> list[str]:
    """Tokens a TTS engine is likely to misread. Used by evals as a hard check."""
    return re.findall(r"\d|[₹$%#@&*_`{}|]|\b[A-Z]{2,5}[-\u2010-\u2014]\d+", text)


class SentenceBuffer:
    """LLM text arrives in token fragments; '₹1,2' + '4,500' cannot be normalised
    piecewise. Buffer to sentence boundaries, normalise whole sentences, and flush
    the remainder at the end. Costs one sentence of latency, never more."""

    def __init__(self, **norm_kwargs) -> None:
        self._buf = ""
        self._kw = norm_kwargs

    def push(self, fragment: str) -> list[str]:
        self._buf += fragment
        out = []
        while (m := _SENTENCE_END.search(self._buf)):
            sentence, self._buf = self._buf[: m.end()], self._buf[m.end():]
            out.append(normalize_for_tts(sentence, **self._kw))
        return out

    def flush(self) -> list[str]:
        rest, self._buf = self._buf, ""
        return [normalize_for_tts(rest, **self._kw)] if rest.strip() else []
