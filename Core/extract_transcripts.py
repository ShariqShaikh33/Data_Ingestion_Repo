#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_transcripts_gemma.py  (v3 - LLM + structural extraction)

Hybrid extractor for Project Saksham call transcripts. 
Uses local Gemma3:12b (via Ollama) as the primary extraction engine for 
Degree, Specialisation, Interest, and Experience. Falls back to deterministic 
keyword/sentence-structure extraction if the LLM fails.


"""

from __future__ import annotations
import argparse
import logging
import re
import unicodedata
import json
from collections import OrderedDict
from pathlib import Path
import urllib.request

import pandas as pd

try:
    from rapidfuzz import fuzz
    _HAVE_RAPIDFUZZ = True
except Exception:
    _HAVE_RAPIDFUZZ = False

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("extract_transcripts")

# --------------------------------------------------------------------------- #
# CONFIG & LLM CLIENT SETUP
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# CAMPAIGN CONFIG LOADER
# --------------------------------------------------------------------------- #
# The engine is CAMPAIGN-BLIND: all campaign-specific values (model, prompt,
# keyword dictionaries, output columns, field mapping) are loaded from a YAML
# config file chosen with --campaign. The guardrail LOGIC stays in this engine
# and operates on whatever keyword lists the config provides.
import yaml as _yaml

HIGH, MEDIUM, LOW = 0.95, 0.75, 0.45

# These module-level globals are POPULATED by load_campaign_config(). They keep
# the same names the guardrail functions already use, so that logic is unchanged.
CFG = {}
OLLAMA_MODEL = None
OLLAMA_URL = None
GEMMA_SYSTEM_PROMPT = None
GEMMA_PROMPT_TEMPLATE = None
GEMMA_OUTPUT_KEYS = []
GEMMA_TEMPERATURE = 0
GEMMA_TIMEOUT = 300

TRANSCRIPT_COL_CANDIDATES = []
SUMMARY_COL_CANDIDATES = []
DEGREE_COL_CANDIDATES = []
CERT_FLAG_COL_CANDIDATES = []
CERT_NAME_COL_CANDIDATES = []

DEGREE_KEYWORDS = OrderedDict()
SPECIALIZATION_KEYWORDS = OrderedDict()
INTEREST_FIELD_KEYWORDS = OrderedDict()
CERTIFICATE_KEYWORDS = OrderedDict()
KNOWN_EMPLOYERS = []
NON_EMPLOYER_WORDS = set()

FIELD_MAPPING = {}
OUTPUT_COLUMN_ORDER = []
NEW_COLUMNS = []

MIN_CONTENT_CHARS = 15
PLACEHOLDER_TEXTS = {
    "no summary available", "no transcript available", "not available",
    "n/a", "na", "none", "no data", "no response", "-", "--",
}

# NON_EMPLOYER_WORDS may be extended by config; keep a base fallback so the
# employer guardrail still works even if a config omits it.
_BASE_NON_EMPLOYER_WORDS = {
    "field", "domain", "sector", "government", "private", "job", "work",
    "company", "manager", "team", "process", "role", "area", "same",
}


def _find_config_file(campaign: str) -> Path:
    """Locate <campaign>.yaml next to this script, in ./configs, or ./campaigns."""
    here = Path(__file__).resolve().parent
    for cand in [here / f"{campaign}.yaml",
                 here / "configs" / f"{campaign}.yaml",
                 here / "campaigns" / f"{campaign}.yaml",
                 Path(f"{campaign}.yaml")]:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Config for campaign '{campaign}' not found. Looked for "
        f"'{campaign}.yaml' next to the script, in ./configs/, and ./campaigns/."
    )


def load_campaign_config(campaign: str):
    """Load a campaign YAML and populate all engine globals from it."""
    global CFG, OLLAMA_MODEL, OLLAMA_URL, GEMMA_SYSTEM_PROMPT, GEMMA_PROMPT_TEMPLATE
    global GEMMA_OUTPUT_KEYS, GEMMA_TEMPERATURE, GEMMA_TIMEOUT
    global TRANSCRIPT_COL_CANDIDATES, SUMMARY_COL_CANDIDATES, DEGREE_COL_CANDIDATES
    global CERT_FLAG_COL_CANDIDATES, CERT_NAME_COL_CANDIDATES
    global DEGREE_KEYWORDS, SPECIALIZATION_KEYWORDS, INTEREST_FIELD_KEYWORDS
    global CERTIFICATE_KEYWORDS, KNOWN_EMPLOYERS, NON_EMPLOYER_WORDS
    global FIELD_MAPPING, OUTPUT_COLUMN_ORDER, NEW_COLUMNS
    global MIN_CONTENT_CHARS, PLACEHOLDER_TEXTS

    path = _find_config_file(campaign)
    with open(path, "r", encoding="utf-8") as f:
        CFG = _yaml.safe_load(f)
    log.info("Loaded campaign config: %s", path.name)

    m = CFG.get("model", {})
    OLLAMA_MODEL = m.get("ollama_model", "gemma3:12b")
    OLLAMA_URL = m.get("ollama_url", "http://127.0.0.1:11434/api/chat")
    GEMMA_TEMPERATURE = m.get("temperature", 0)
    GEMMA_TIMEOUT = m.get("timeout_seconds", 300)

    GEMMA_SYSTEM_PROMPT = CFG.get("gemma_system_prompt",
        "You are a precise data extraction analyst. Output valid JSON only.")
    GEMMA_PROMPT_TEMPLATE = CFG.get("gemma_prompt", "")
    GEMMA_OUTPUT_KEYS = CFG.get("gemma_output_keys", [])

    ic = CFG.get("input_columns", {})
    TRANSCRIPT_COL_CANDIDATES = ic.get("transcript_candidates", ["transcript"])
    SUMMARY_COL_CANDIDATES = ic.get("summary_candidates", ["summary"])
    DEGREE_COL_CANDIDATES = ic.get("degree_candidates", ["degree"])
    CERT_FLAG_COL_CANDIDATES = ic.get("cert_flag_candidates", ["any skill certificate"])
    CERT_NAME_COL_CANDIDATES = ic.get("cert_name_candidates", ["name of skill certificate"])

    # keyword dicts (preserve insertion order from YAML -> priority order)
    DEGREE_KEYWORDS = OrderedDict(CFG.get("degree_keywords", {}))
    SPECIALIZATION_KEYWORDS = OrderedDict(CFG.get("specialization_keywords", {}))
    INTEREST_FIELD_KEYWORDS = OrderedDict(CFG.get("interest_field_keywords", {}))
    CERTIFICATE_KEYWORDS = OrderedDict(CFG.get("certificate_keywords", {}))
    KNOWN_EMPLOYERS = [str(e).lower() for e in CFG.get("known_employers", [])]
    NON_EMPLOYER_WORDS = set(w.lower() for w in CFG.get("non_employer_words", [])) | _BASE_NON_EMPLOYER_WORDS

    FIELD_MAPPING = CFG.get("field_mapping", {})
    OUTPUT_COLUMN_ORDER = CFG.get("output_column_order", [])
    # NEW_COLUMNS = the extracted columns (values from field_mapping) + confidence cols
    NEW_COLUMNS = list(dict.fromkeys(list(FIELD_MAPPING.values()) +
                                     ["Degree Confidence", "Extraction Confidence"]))

    g = CFG.get("guardrails", {})
    MIN_CONTENT_CHARS = g.get("min_content_chars", 15)
    if g.get("placeholder_texts"):
        PLACEHOLDER_TEXTS = set(str(p).lower() for p in g["placeholder_texts"])

    return CFG

# STOPWORD_VALUES and NEGATION_CUES are generic (not campaign-specific) and stay
# in the engine. All other keyword dicts are loaded from the campaign config.
STOPWORD_VALUES = {
    "", "the", "a", "an", "same", "this", "that", "any", "some", "no", "not",
    "as", "is", "in", "of", "to", "at", "on", "for", "was", "are", "be",
    "wrong number", "network problem", "quick", "one", "initial introduction",
    "skill census", "specific role", "specific", "role in mind", "field",
    "particular", "sure", "yet", "poor audio quality", "hello", "customer",
    "assistant", "call", "profile", "information", "maharashtra government",
    "his", "her", "their", "same field", "subject", "degree", "branch",
}

NEGATION_CUES = ["not ", "no ", "n't", "never", "nahi", "nahin", "nako",
                 "without", "bina", "didn't", "does not", "do not"]

# --------------------------------------------------------------------------- #
# TEXT UTILITIES
# --------------------------------------------------------------------------- #
def clean_text(text) -> str:
    if text is None:
        return ""
    s = str(text)
    if s.strip().lower() in ("nan", "none", "nat"):
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u200b", " ").replace("\ufeff", " ")
    s = re.sub(r'[""]', '"', s)
    s = re.sub(r"['']", "'", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    s = s.strip()
    if s.lower() in PLACEHOLDER_TEXTS:
        return ""
    return s

def _is_blank(v) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return str(v).strip().lower() in ("", "nan", "none", "nat")

def _clean_value(v: str) -> str:
    if not v:
        return ""
    v = v.strip(" .,-\"'\n\t")
    v = re.sub(r"\s+", " ", v)
    v = re.split(r"\b(and expressed|and mentioned|and shared|and has|however|though|but )", v)[0].strip(" .,-")
    v = re.sub(r"\s+(branch|field|degree|stream|subject|course)\s*$", "", v, flags=re.IGNORECASE).strip(" .,-")
    low = v.lower()
    if low in STOPWORD_VALUES or len(v) < 2:
        return ""
    if re.match(r"^(or|and|the|a|an|of|in|to|for|with|as|is|was|his|her|their)\b", low):
        return ""
    noise = [
        "captur", "obtain", "collect", "provide", "record", "missing", "requested",
        "understand", "question", "repeat", "confus", "expressed", "mentioned",
        "confirmed", "asked", "however", "despite", "attempt", "did not", "was not",
        "were not", "could", "should", "would", "remains", "needs", "still",
        "poor audio", "the assistant", "the customer", "the user", "profile",
        "conversation", "interaction", "brief", "weak", "unclear", "inquire",
        "future work", "main subject", "information", "detail", "complete",
        "response", "identity", "purpose", "regarding", "initiated",
        "proxy", "result", "wrong number", "quick", "one time", "part of",
        "senior customer executive or", "follow up", "member", "minute",
    ]
    if any(w in low for w in noise):
        return ""
    if len(v) > 35 or len(v.split()) > 4:
        return ""
    return v.strip(" .,-")

def _has_negation(hay: str, start: int, window: int = 22) -> bool:
    left = hay[max(0, start - window):start].lower()
    return any(cue in left for cue in NEGATION_CUES)

def user_turns(transcript: str) -> str:
    picked = []
    for ln in transcript.splitlines():
        m = re.match(r"^\s*(user|candidate)\s*[:\-]\s*(.*)$", ln, re.IGNORECASE)
        if m:
            picked.append(m.group(2))
    return "\n".join(picked) if picked else transcript

# --------------------------------------------------------------------------- #
# FALLBACK DIALOGUE PARSING & DETERMINISTIC LOGIC
# --------------------------------------------------------------------------- #
# (Preserving deterministic functions for when LLM fails or returns 'Not Mentioned')
CONFIRM_WORDS = ["yes", "yeah", "yep", "correct", "right", "exactly", "that's right", "thats right", "confirm", "confirmed", "sure", "ok", "okay", "हाँ", "हां", "हाँ जी", "हां जी", "जी", "जी हाँ", "सही", "बिलकुल", "बिल्कुल", "ठीक", "सही है", "हो", "होय", "बरोबर", "हो जी", "haan", "han", "haanji", "haan ji", "bilkul", "bilkul sahi", "sahi", "sahi hai", "ho", "hoy", "barobar", "theek", "thik", "thik hai"]
REJECT_WORDS = ["no", "nope", "wrong", "incorrect", "not correct", "that's wrong", "thats wrong", "नहीं", "नही", "गलत", "गलत है", "नाही", "चुकीचं", "चुकीचे", "नको", "nahi", "nahin", "nako", "galat", "galat hai", "wrong hai"]

def parse_turns(transcript: str):
    turns = []
    for ln in transcript.splitlines():
        m = re.match(r"^\s*(assistant|bot|agent|पूछने वाला|सहायक|user|candidate|उपयोगकर्ता)\s*[:\-]\s*(.*)$", ln, re.IGNORECASE)
        if m:
            sp = m.group(1).lower()
            speaker = "user" if sp in ("user", "candidate", "उपयोगकर्ता") else "assistant"
            turns.append([speaker, m.group(2).strip()])
        elif turns:
            turns[-1][1] += " " + ln.strip()
    return turns

def _response_polarity(user_text: str):
    low = " " + user_text.lower().strip() + " "
    for w in REJECT_WORDS:
        if re.search(r"(?<![a-zA-Z])" + re.escape(w) + r"(?![a-zA-Z])", low):
            return "reject"
    for w in CONFIRM_WORDS:
        if re.search(r"(?<![a-zA-Z])" + re.escape(w) + r"(?![a-zA-Z])", low):
            return "confirm"
    return "neutral"

def _degree_negated(low: str, pos: int, token: str) -> bool:
    before = low[max(0, pos - 25):pos]
    after = low[pos + len(token):pos + len(token) + 25]
    neg = ["nahi", "nahin", "नहीं", "नही", "nako", "नको", "galat", "गलत", "wrong",
           "नाही", "चुकीच", "not"]
    # 'after' negation: only the immediate clause (stop at '.'/','/'but').
    after_clause = re.split(r"\.|,|\bbut\b", after)[0]
    if any(w in after_clause for w in neg):
        return True
    # 'before' negation: back to the previous clause boundary only (period, comma,
    # or Hindi/Marathi completion verb हुआ/केलं). Prevents "not a B.Com. He has
    # done B.Tech" or "BCom nahi hua, BTech hua" from negating the second degree.
    before_clause = re.split(r"\.|,|हुआ|केलं|केला|आहे", before)[-1]
    if any(w in before_clause for w in ["not ", "n't ", "no ", "nahi", "नहीं", "नाही", "नही"]):
        return True
    return False

DEGREE_TOKEN = (r"\b(ph\.?d|mba|m\.?\s?com|m\.?\s?sc|m\.?\s?a|mca|m\.?\s?tech|"
                r"b\.?\s?tech|b\.?\s?com|b\.?\s?sc|bca|bba|b\.?\s?a|diploma|iti|"
                r"hsc|ssc|12th|10th|graduate|graduation|"
                r"engineering degree|engineering|arts degree|arts)\b")
FALSE_DEGREE_CONTEXTS = ["ma'am", "maam", "madam", "ma am"]

def _all_degrees_in(text: str):
    low = text.lower()
    # Collect every candidate match WITH its (start, end) span so that
    # overlapping matches can be resolved afterwards (longest span wins).
    # This prevents a shorter degree token that is a substring of a longer one
    # from being emitted as a spurious second degree. The classic case is the
    # Devanagari "बीएससी" (BSc): the substring "बीए" (BA) sits at its start, so
    # a naive find() would wrongly also register BA — and because callers take
    # the LAST affirmed degree, BA could override the real BSc.
    raw_matches = []  # (canon, affirmed, start, length)
    for m in re.finditer(DEGREE_TOKEN, low):
        tok = m.group(1)
        start = m.start()
        ctx = low[max(0, start - 2):start + len(tok) + 4]
        if any(fc in ctx for fc in FALSE_DEGREE_CONTEXTS):
            continue
        if tok in ("ma", "ba"):
            near = low[max(0, start - 30):start + 30]
            if not any(cue in near for cue in ["degree", "graduat", "complet", "किया", "पूरा", "पास", "बी.ए", "एम.ए", "arts", "बीए", "एमए", "b.a", "m.a"]):
                continue
        raw = tok.replace(".", "").replace(" ", "")
        canon = ""
        for c, forms in DEGREE_KEYWORDS.items():
            if raw in [f.replace(".", "").replace(" ", "").strip() for f in forms]:
                canon = c
                break
        if not canon:
            for c, forms in DEGREE_KEYWORDS.items():
                for f in forms:
                    ff = f.replace(".", "").replace(" ", "").strip()
                    if ff and (raw.startswith(ff) or ff.startswith(raw)):
                        canon = c
                        break
                if canon: break
        if not canon:
            canon = m.group(1).upper()
        affirmed = not _degree_negated(low, start, m.group(1))
        raw_matches.append((canon, affirmed, start, len(tok)))
    for canon, forms in DEGREE_KEYWORDS.items():
        for f in forms:
            if re.search(r"[\u0900-\u097F]", f):
                # find EVERY occurrence of this Devanagari form, not just the first
                search_start = 0
                while True:
                    idx = low.find(f, search_start)
                    if idx == -1:
                        break
                    affirmed = not _degree_negated(low, idx, f)
                    raw_matches.append((canon, affirmed, idx, len(f)))
                    search_start = idx + 1

    # --- Resolve overlaps: longest match wins ---
    # Sort longest-first so a longer span always gets first claim on its range.
    raw_matches.sort(key=lambda t: -t[3])
    kept = []
    kept_spans = []  # (start, end) already claimed
    for canon, affirmed, start, length in raw_matches:
        end = start + length
        overlaps = any(start < k_end and end > k_start
                       for (k_start, k_end) in kept_spans)
        if overlaps:
            continue
        kept.append((canon, affirmed, start))
        kept_spans.append((start, end))
    # Restore original left-to-right order so the "last affirmed" logic in
    # _degree_in / _degree_from_summary (which picks the last mention) is preserved.
    kept.sort(key=lambda t: t[2])
    return [(canon, affirmed) for canon, affirmed, _ in kept]

def _degree_in(text: str):
    all_d = _all_degrees_in(text)
    affirmed = [c for c, ok in all_d if ok]
    if affirmed: return affirmed[-1]
    return ""

def _is_substantive_answer(text: str) -> bool:
    low = text.lower().strip()
    if len(low) < 2: return False
    confusion = ["समझ नहीं", "समझ नई", "समझ नाही", "नहीं आ रहा", "नई आ रहा", "क्या बोल", "काय बोल", "don't understand", "didn't understand", "not understand", "confus", "hello", "आवाज नहीं", "sunai nahi", "network", "आवाज", "repeat", "फिर से", "पुन्हा", "कळत नाही", "समजत नाही", "pardon", "continue", "आगे बढ़", "wrong number", "कोण", "who is this"]
    if any(c in low for c in confusion): return False
    if not re.search(r"[a-z\u0900-\u097F]", low): return False
    return True

def extract_degree_dialogue(transcript: str, summary: str):
    turns = parse_turns(transcript)
    if not turns:
        return "", 0.0
    pending, committed, asked_subject = "", "", False
    for i, (speaker, text) in enumerate(turns):
        if speaker == "assistant":
            d = _degree_in(text)
            if d: pending = d
            tl = text.lower()
            if any(cue in tl for cue in ["subject", "विषय", "सब्जेक्ट", "branch", "ब्रांच", "कौन सा", "which subject"]):
                asked_subject = True
        else:
            user_degree = _degree_in(text)
            polarity = _response_polarity(text)
            if user_degree:
                committed, pending, asked_subject = user_degree, "", False
                continue
            if polarity == "confirm" and pending:
                committed, pending, asked_subject = pending, "", False
                continue
            if polarity == "reject":
                pending, asked_subject = "", False
                continue
            if pending and asked_subject and _is_substantive_answer(text):
                committed, pending, asked_subject = pending, "", False
                continue
    if committed: return committed, HIGH
    return "", 0.0

def _degree_from_summary(summary: str):
    if not summary:
        return "", 0.0
    low = summary.lower()
    # Find all degrees with affirmed/negated status.
    all_d = _all_degrees_in(low)
    affirmed = [canon for canon, ok in all_d if ok]
    if not affirmed:
        return "", 0.0
    # The summary is an AI description. Accept an affirmed degree only when the
    # summary shows the CANDIDATE established it — i.e. a confirmation cue appears
    # somewhere. This lets "missing information ... she confirmed ... graduate"
    # still work (candidate confirmed), while a pure bot-only note like
    # "records showed B.Com, subject pending" (no candidate cue) is rejected.
    confirm_cues = ["clarified", "confirmed", "stated", "said", "completed",
                    "has a", "has an", "has done", "holds", "degree is", "degree as",
                    "is a graduate", "she is", "he is", "pursued", "studied",
                    "mentioned", "told"]
    # Hard bot-only signals: if the summary ONLY describes unverified records / no
    # contact and has NO confirmation cue, reject.
    bot_only = ["records showed", "records indicated", "no response", "could not",
                "did not provide", "unable to", "did not confirm", "was not confirmed",
                "profile incomplete", "further contact needed"]
    has_confirm = any(cue in low for cue in confirm_cues)
    has_bot_only = any(b in low for b in bot_only)
    if not has_confirm and has_bot_only:
        return "", 0.0
    if not has_confirm:
        # no explicit confirmation language at all -> be conservative, reject
        return "", 0.0
    # take the LAST affirmed degree (handles "not a B.Com ... engineering" -> BE/BTech,
    # and "B.Com degree [bot note] ... graduate [confirmed]" -> Graduate)
    return affirmed[-1], HIGH

def _find_keyword(hay_low: str, surface: str) -> int:
    surface = surface.lower().strip()
    idx = hay_low.find(surface)
    if idx != -1: return idx
    if _HAVE_RAPIDFUZZ and len(surface) >= 6 and re.fullmatch(r"[a-z ]+", surface):
        for token in set(re.findall(r"[a-zA-Z]{5,}", hay_low)):
            if fuzz.ratio(token, surface) >= 92: return hay_low.find(token)
    return -1

def _match_dict(text_low: str, dictionary):
    for canon, forms in dictionary.items():
        for f in forms:
            idx = _find_keyword(text_low, f.strip())
            if idx != -1 and not _has_negation(text_low, idx): return canon, HIGH
    return "", 0.0

SUBJECT_PATTERNS = [r"['\"]?([A-Za-z0-9.&/\- ]{2,30}?)['\"]?\s+as\s+(?:their|her|his|the)\s+(?:main\s+)?subject", r"subject\s+(?:as|is|was)\s+([A-Za-z0-9.&/\- ]{2,40})", r"subject\s*[:\-]\s*([A-Za-z0-9.&/\- ]{2,40})", r"(?:degree|b\.?tech|b\.?com|m\.?com|mba|b\.?sc|diploma|iti)\s+(?:in|subject|branch)\s+([A-Za-z0-9.&/\- ]{2,40})", r"branch\s+(?:as|is|was|of)?\s*([A-Za-z0-9.&/\- ]{2,40})", r"speciali[sz]ation\s+(?:in|as|is)?\s*([A-Za-z0-9.&/\- ]{2,40})", r"(?:main\s+)?subjects?\s+([A-Za-z0-9.&/,\- ]{2,40})"]
FIELD_PATTERNS = [r"interest(?:ed)?\s+in\s+(?:working\s+in\s+)?(?:the\s+)?([A-Za-z /&]{3,30}?)\s+field", r"want(?:s|ed)?\s+to\s+work\s+in\s+(?:the\s+)?([A-Za-z /&]{3,30}?)\s+field", r"work\s+in\s+(?:the\s+)?([A-Za-z /&]{3,30}?)\s+field", r"career\s+(?:interest|goal)\s+(?:in|is)\s+([A-Za-z /&]{3,30})"]
ROLE_PATTERNS = [r"\bas\s+an?\s+([A-Za-z ]{3,30}?)(?:\s+role|\.|,|;| and | with | in )", r"role\s+(?:as|of|is)\s+(?:an?\s+)?([A-Za-z ]{3,30})", r"(?:specifically|particularly)\s+(?:an?\s+)?([A-Za-z ]{3,30}?)\s+role"]

def _first_capture(patterns, text) -> str:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = _clean_value(m.group(1))
            if val: return val
    return ""

def extract_specialization(text: str):
    low = text.lower()
    # If the summary explicitly says the subject/specialisation was NOT captured,
    # do not extract one (prevents pulling the subject from the bot's question or
    # the degree name when the candidate never actually gave it).
    not_captured = [
        "could not be captured", "could not be obtained", "not be captured",
        "unclear response", "not provide", "did not provide", "not captured",
        "unable to capture", "no subject", "subject details could not",
        "does not have a specialisation", "does not have a specialization",
        "no specialisation", "no specialization", "not confirm the subject",
        "subject was not", "audio issue", "further contact is needed",
    ]
    if any(p in low for p in not_captured):
        return "", 0.0
    # Remove degree-name phrases so "Bachelor of Science" doesn't yield subject
    # "Science", and "B.Com degree" doesn't yield "Commerce" from the degree word.
    cleaned = low
    for dn in ["bachelor of science", "bachelor of arts", "bachelor of commerce",
               "bachelor of technology", "bachelor of engineering", "b.sc degree",
               "b.com degree", "b.a degree", "b.tech degree", "bachelor of computer",
               "arts degree", "science degree", "commerce degree", "engineering degree",
               "her arts", "his arts", "arts and", "graduate degree"]:
        cleaned = cleaned.replace(dn, " ")
    val = _first_capture(SUBJECT_PATTERNS, cleaned)
    if val: return val, HIGH
    return _match_dict(cleaned, SPECIALIZATION_KEYWORDS)


def extract_specialization_dialogue(transcript: str, summary: str):
    """Specialisation from BOTH transcript and summary (union of both sources)."""
    return _union(extract_specialization(summary or ""),
                  extract_specialization(transcript or ""))

def extract_interest(text: str):
    parts = []
    field = _first_capture(FIELD_PATTERNS, text)
    if field and field.lower() not in STOPWORD_VALUES: parts.append(field.title() + " field")
    role = _first_capture(ROLE_PATTERNS, text)
    if role and role.lower() not in STOPWORD_VALUES: parts.append(role.title())
    if not parts:
        canon, conf = _match_dict(text.lower(), INTEREST_FIELD_KEYWORDS)
        if canon: parts.append(canon)
    parts = list(OrderedDict.fromkeys(parts))
    deduped = []
    for p in parts:
        pl = p.lower().replace(" field", "").strip()
        if not any(pl != q.lower().replace(" field", "").strip() and pl in q.lower() for q in parts):
            deduped.append(p)
    return ", ".join(deduped), (HIGH if deduped else 0.0)


def extract_interest_dialogue(transcript: str, summary: str):
    """Interest from BOTH transcript and summary (union of both sources)."""
    return _union(extract_interest(summary or ""),
                  extract_interest(transcript or ""))

def extract_certificate(text: str):
    low = text.lower()
    names = []
    for canon, forms in CERTIFICATE_KEYWORDS.items():
        for f in forms:
            idx = _find_keyword(low, f.strip())
            if idx != -1 and not _has_negation(low, idx):
                names.append(canon)
                break
    names = list(OrderedDict.fromkeys(names))
    return ("Yes" if names else ""), ", ".join(names), (HIGH if names else 0.0)

def _looks_like_field_not_employer(candidate: str) -> bool:
    c = candidate.strip().lower()
    if not c: return True
    return any(c == w or w in c for w in NON_EMPLOYER_WORDS)

def extract_employer(text: str):
    low = text.lower()
    for emp in KNOWN_EMPLOYERS:
        idx = low.find(emp)
        if idx != -1 and not _has_negation(low, idx): return emp.title(), HIGH
    for pat in [r"\bwork(?:ing)?\s+(?:at|for|with)\s+([A-Z][A-Za-z0-9&.\- ]{2,40})", r"\bemployed\s+(?:at|by|with)\s+([A-Z][A-Za-z0-9&.\- ]{2,40})", r"\bcompany\s+(?:is|name is|called)\s+([A-Z][A-Za-z0-9&.\- ]{2,40})"]:
        m = re.search(pat, text)
        if m:
            cand = _clean_value(m.group(1))
            if cand and not _looks_like_field_not_employer(cand): return cand, MEDIUM
    return "", 0.0

def extract_additional_information(summary: str):
    """Extract ONLY the candidate's important, relevant facts from the SUMMARY.

    Keeps things like: work experience, current employment status, availability,
    family/financial situation, aspirations. Drops: bot narration, process/status
    notes, greetings, degree/subject restatement (those live in their own columns),
    and anything not about the candidate.
    """
    if not summary:
        return "", 0.0
    summary = clean_text(summary)
    if not summary:
        return "", 0.0
    low_all = summary.lower()

    # If the whole summary is a non-call (voicemail, no response, inconclusive), nothing.
    dead_call = ["no summary available", "no response", "voicemail", "answering machine",
                 "at the tone", "record your message", "inconclusive", "no conversation",
                 "did not respond", "call was cut short", "no further conversation"]
    if any(p in low_all for p in dead_call):
        return "", 0.0

    facts = []

    # 1) Work experience (years / months / fresher) — a key candidate fact.
    m = re.search(r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|half)"
                  r"[\-\s]*(?:and a half\s*)?(?:years?|yrs?|months?|month)"
                  r"(?:\s+of)?\s*(?:work\s+)?(?:experience|exp)?", low_all)
    if m:
        exp = m.group(0).strip()
        exp = re.sub(r"\s+(of|work|,)\s*$", "", exp).strip()  # trim trailing filler
        facts.append("Experience: " + exp)
    elif re.search(r"\bfresher\b|no (?:prior |work )?experience|कोई अनुभव नहीं", low_all):
        facts.append("Fresher (no experience)")

    # 2) Employment status the candidate stated about themselves.
    status_cues = {
        "currently working / employed": ["currently working", "presently working",
                                          "has a company job", "is working", "already working",
                                          "employed at", "works at", "doing a job", "job currently"],
        "currently studying": ["still studying", "pursuing", "final year", "is a student"],
        "self-employed / business": ["own business", "self employed", "self-employed",
                                     "runs a", "has a shop"],
        "unemployed / seeking work": ["not working", "unemployed", "looking for work",
                                      "seeking a job", "wants a job", "needs a job"],
    }
    for label, cues in status_cues.items():
        if any(c in low_all for c in cues):
            facts.append(label)
            break

    # 3) Situational / personal candidate facts worth keeping.
    situational = {
        "Supporting family / child": ["support her child", "support his family",
                                       "support their family", "family responsibility"],
        "Financial difficulty": ["financial problem", "financial difficulty",
                                  "money problem", "financially"],
        "Available weekends only": ["only on weekend", "weekends only", "weekend only"],
        "Available evenings only": ["only in the evening", "evenings only", "after work"],
        "Wants placement / job support": ["wants placement", "looking for placement",
                                          "job placement", "wants a job opportunity"],
        "Willing to relocate": ["willing to relocate", "can relocate", "ready to move"],
        "Not willing to relocate": ["cannot relocate", "not willing to relocate",
                                    "won't relocate"],
        "Answered by family member (proxy)": ["father confirmed", "mother confirmed",
                                              "spoke to his father", "spoke to her father",
                                              "family member", "acting as a proxy",
                                              "proxy"],
    }
    for label, cues in situational.items():
        if any(c in low_all for c in cues):
            facts.append(label)

    facts = list(OrderedDict.fromkeys([f for f in facts if f]))
    result = ", ".join(facts)
    return result, (HIGH if result else 0.0)

def confidence_score(*cs) -> float:
    vals = [c for c in cs if isinstance(c, (int, float))]
    return max(vals) if vals else 0.0

def _pick(a, b):
    (va, ca), (vb, cb) = a, b
    if va and vb: return (va, ca) if ca >= cb else (vb, cb)
    return (va, ca) if va else (vb, cb)

def _union(a, b):
    (va, ca), (vb, cb) = a, b
    parts = []
    for v in (va, vb):
        if v: parts += [p.strip() for p in v.split(",") if p.strip()]
    parts = list(OrderedDict.fromkeys(parts))
    return (", ".join(parts), max(ca, cb)) if parts else ("", 0.0)

def _dedupe_substrings(value: str) -> str:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    def key(s): return s.lower().replace(" & ", " and ").replace("  ", " ").strip()
    seen, uniq = {}, []
    for p in parts:
        k = key(p)
        if k not in seen:
            seen[k] = p
            uniq.append(p)
    out = []
    for p in uniq:
        pl = key(p)
        if not any(pl != key(q) and pl in key(q) for q in uniq):
            out.append(p)
    return ", ".join(out)

# --------------------------------------------------------------------------- #
# NEW: LLM INTEGRATION
# --------------------------------------------------------------------------- #
def call_gemma_llm(transcript: str, summary: str) -> dict:
    """
    Send transcript + summary to the Gemma model running through Ollama.
    """

    # Build the prompt from the campaign config's template.
    prompt = GEMMA_PROMPT_TEMPLATE.format(transcript=transcript or "(empty)",
                                          summary=summary or "(empty)")

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": GEMMA_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": GEMMA_TEMPERATURE
            }
        }

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=GEMMA_TIMEOUT) as response:
            response_data = json.loads(response.read().decode("utf-8"))

        output = response_data.get("message", {}).get("content", "").strip()

        if not output:
            log.error("Ollama returned an empty response.")
            return {}

        return json.loads(output)

    except Exception as e:
        log.error(f"Ollama extraction error: {e}")
        return {}

def extract_row(text: str, summ: str) -> dict:
    T = text or ""
    S = summ or ""
    ut = user_turns(T)
    
    # 1. Ask Gemma LLM First
    gemma_data = call_gemma_llm(T, S)
    g_deg = gemma_data.get("degree", "Not Mentioned")
    g_spec = gemma_data.get("specialization", "Not Mentioned")
    g_int = gemma_data.get("interest", "Not Mentioned")
    g_exp = gemma_data.get("experience", "Not Mentioned")
    
    # 2. Assign Degree (Gemma Priority -> Fallback to Regex)
    # 2. VALIDATE DEGREE USING ACTUAL CANDIDATE DIALOGUE
#
# Gemma is allowed to interpret the conversation, but it is NOT
# allowed to override the candidate-evidence rule.
#
# The degree is accepted ONLY when:
#   1. Candidate explicitly states a degree, OR
#   2. Candidate confirms a degree previously stated by assistant, OR
#   3. Candidate corrects the assistant with another degree, OR
#   4. Candidate answers the assistant's degree/subject question
#      in a substantive way after the assistant stated the degree.
#
# The Summary is NEVER allowed to establish a degree.

    # DEGREE — candidate-confirmation rule, checked in BOTH transcript and summary.
    # Priority: transcript dialogue evidence first; if none, fall back to the
    # summary (only when the summary describes the CANDIDATE confirming/stating/
    # correcting a degree — bot-only "records show" phrasing is rejected).
    validated_deg = extract_degree_dialogue(T, "")
    if validated_deg[0]:
        deg = validated_deg
    else:
        summary_deg = _degree_from_summary(S)
        if summary_deg[0]:
            deg = summary_deg
        else:
            # No candidate-confirmed degree in either column.
            # Reject even if Gemma proposed one.
            deg = ("", 0.0)
        
    # 3. Specialization: Gemma first; if Gemma has nothing, fall back to BOTH-column rules
    if g_spec and g_spec.lower() != "not mentioned":
        spec = (g_spec, HIGH)
    else:
        spec = extract_specialization_dialogue(T, S)

    # 4. Interest: Gemma first; if Gemma has nothing, fall back to BOTH-column rules
    if g_int and g_int.lower() != "not mentioned":
        interest = (g_int, HIGH)
    else:
        interest = extract_interest_dialogue(T, S)
        
    # Certificates and Employer (Regex Fallback remains)
    cert_s = extract_certificate(S)
    cert_t = extract_certificate(T)
    cert_names = _union((cert_s[1], cert_s[2]), (cert_t[1], cert_t[2]))
    cert_has = "Yes" if ((cert_s[0] == "Yes" or cert_t[0] == "Yes") and cert_names[0]) else ""
    employer = _pick(extract_employer(S), extract_employer(T))
    
    # 5. Additional Information — a REFINED version of the summary produced by Gemma:
    # the summary rewritten into clean prose about the candidate, with all bot/
    # process narration removed. Falls back to the rule-based fact extractor only
    # if Gemma didn't return a refined summary (e.g. Ollama unavailable).
    g_refined = gemma_data.get("refined_summary", "")
    g_refined = (g_refined or "").strip()
    if g_refined and g_refined.lower() not in ("not mentioned", "none", "null", "n/a"):
        add = (g_refined, HIGH)
    else:
        add = extract_additional_information(S)   # rule-based fallback

    return {"degree": deg, "spec": spec, "interest": interest,
            "cert_has": cert_has, "cert_names": cert_names,
            "employer": employer, "additional": add}

# --------------------------------------------------------------------------- #
# COLUMN DETECTION / IO
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return str(s).strip().lower().replace("_", " ")

def detect_column(df, candidates, fuzzy_min=90):
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        if _norm(cand) in norm_map: return norm_map[_norm(cand)]
    if _HAVE_RAPIDFUZZ:
        for cand in candidates:
            for ncol, real in norm_map.items():
                if fuzz.ratio(_norm(cand), ncol) >= fuzzy_min: return real
    return None

def detect_transcript_column(df):
    col = detect_column(df, TRANSCRIPT_COL_CANDIDATES)
    if col: return col
    # don't let the SUMMARY column be auto-picked as the transcript
    summary_col = detect_column(df, SUMMARY_COL_CANDIDATES)
    best, best_len = None, 0
    for c in df.columns:
        if c == summary_col:
            continue
        series = df[c].dropna().astype(str)
        if series.empty: continue
        avg = series.map(len).mean()
        if avg > best_len: best, best_len = c, avg
    if best is not None and best_len > 30:
        log.info("Transcript column auto-detected by content: %r", best)
        return best
    return None

def load_excel(path):
    log.info("Reading %s ...", path.name)
    df = pd.read_excel(path, dtype=str, engine="openpyxl")
    df = df.where(pd.notna(df), None)
    log.info("  %d rows, %d columns", len(df), len(df.columns))
    return df

def save_output(df, out_path):
    log.info("Writing %s ...", out_path.name)
    df.to_excel(out_path, index=False, engine="openpyxl")
    log.info("  done (%d rows, %d columns).", len(df), len(df.columns))

# --------------------------------------------------------------------------- #
# OUTPUT COLUMN ORDER & UPDATE LOGIC
# --------------------------------------------------------------------------- #
NEW_COLUMNS = [
    "Degree", "Specialisation / Subjects", "Any Skill Certificate",
    "Name Of Skill Certificate", "Current Employer Name (Real Time)",
    "Current Interest in Jobs / Self Employment / Skills",
    "Additional Information based on AI Calls",
    "Degree Confidence", "Extraction Confidence",
]

OUTPUT_COLUMN_ORDER = [
    "ID", "AGENT ID", "CALLING NUMBER", "PROVIDER NUMBER", "STATUS",
    "TOTAL DURATION", "IS ESCALATED", "IS HOT DEAL", "DETECTED EMOTION",
    "TRANSCRIPT", "SUMMARY",
    "Degree", "Specialisation / Subjects", "Any Skill Certificate",
    "Name Of Skill Certificate", "Current Employer Name (Real Time)",
    "Current Interest in Jobs / Self Employment / Skills",
    "Additional Information based on AI Calls",
    "Degree Confidence", "Extraction Confidence",
    "COMMENTS", "FILE PATH", "CALL STATUS", "DISPOSITION", "DISPOSITION INTENT",
    "Total Duration.1", "Date", "Time", "Call Type", "Next Step Note",
    "Follow Up Required", "Follow Up Datetime", "Campaign Name",
]

def reorder_columns(df, keep_extras=False):
    ordered = [c for c in OUTPUT_COLUMN_ORDER if c in df.columns]
    if keep_extras:
        extras = [c for c in df.columns if c not in ordered]
        return df[ordered + extras]
    return df[ordered]

def update_dataframe(df, transcript_col, summary_col=None, max_rows=None):
    # Resolve output column names from the config's field mapping.
    col_degree  = FIELD_MAPPING.get("degree", "Degree")
    col_spec    = FIELD_MAPPING.get("specialization", "Specialisation / Subjects")
    col_certf   = FIELD_MAPPING.get("any_skill_certificate", "Any Skill Certificate")
    col_certn   = FIELD_MAPPING.get("name_of_skill_certificate", "Name Of Skill Certificate")
    col_emp     = FIELD_MAPPING.get("current_employer", "Current Employer Name (Real Time)")
    col_int     = FIELD_MAPPING.get("interest", "Current Interest in Jobs / Self Employment / Skills")
    col_add     = FIELD_MAPPING.get("additional_information", "Additional Information based on AI Calls")

    # The extracted columns ALWAYS come from extraction. Any same-named column
    # already in the input (e.g. a stale "Degree" = "BCom") is CLEARED first, so
    # the output reflects only what we extracted — never the input's old value.
    extracted_cols = [col_degree, col_spec, col_certf, col_certn, col_emp, col_int,
                      col_add, "Degree Confidence", "Extraction Confidence"]
    for c in extracted_cols:
        df[c] = None   # create if missing, wipe if it already existed in the input

    n = len(df) if max_rows is None else min(max_rows, len(df))
    it = range(n)
    if _HAVE_TQDM: it = tqdm(it, desc="Extracting", unit="row")

    processed = 0
    for i in it:
        ridx = df.index[i]
        text = clean_text(df.at[ridx, transcript_col]) if transcript_col else ""
        summ = clean_text(df.at[ridx, summary_col]) if summary_col else ""

        has_content = (len(text) >= MIN_CONTENT_CHARS) or (len(summ) >= MIN_CONTENT_CHARS)
        if not has_content:
            continue   # extracted cols already cleared to None above

        r = extract_row(text, summ)
        deg_val, deg_conf = r["degree"]
        spec_val, spec_conf = r["spec"]
        int_val, int_conf = r["interest"]
        emp_val, emp_conf = r["employer"]
        cert_names, cert_conf = r["cert_names"]
        add_val, add_conf = r["additional"]

        df.at[ridx, col_degree] = deg_val if deg_val else None
        df.at[ridx, "Degree Confidence"] = str(round(deg_conf, 2)) if deg_val else None
        if spec_val: df.at[ridx, col_spec] = spec_val
        if r["cert_has"] == "Yes": df.at[ridx, col_certf] = "Yes"
        if cert_names: df.at[ridx, col_certn] = cert_names
        if emp_val: df.at[ridx, col_emp] = emp_val
        if int_val: df.at[ridx, col_int] = int_val
        if add_val: df.at[ridx, col_add] = add_val
        df.at[ridx, "Extraction Confidence"] = str(round(
            confidence_score(deg_conf, spec_conf, int_conf, emp_conf, cert_conf, add_conf), 2))
        processed += 1

    log.info("  extracted from %d non-empty row(s).", processed)
    # When --max-rows is set, keep ONLY those rows in the output.
    if max_rows is not None:
        df = df.iloc[:n].copy()
    return df


def process_file(in_path, max_rows=None, keep_extras=False, use_summary=True):
    df = load_excel(in_path)
    tcol = detect_transcript_column(df)
    scol = detect_column(df, SUMMARY_COL_CANDIDATES) if use_summary else None

    # The script works with either or both columns:
    #   - both present  -> read both
    #   - only transcript -> read transcript only
    #   - only summary    -> read summary only
    if tcol is None and scol is None:
        raise RuntimeError(
            "No transcript column AND no summary column found in %s. "
            "At least one is required." % in_path.name)
    if tcol:
        log.info("Using transcript column: %r", tcol)
    else:
        log.info("No transcript column found - using SUMMARY only.")
    if scol:
        log.info("Using summary column: %r", scol)
    elif use_summary:
        log.info("No summary column found - using TRANSCRIPT only.")

    df = update_dataframe(df, tcol, summary_col=scol, max_rows=max_rows)
    df = reorder_columns(df, keep_extras=keep_extras)
    out_path = in_path.with_name(in_path.stem + "_output.xlsx")
    save_output(df, out_path)
    return out_path

def main():
    ap = argparse.ArgumentParser(description="LLM + Offline transcript extractor (campaign-blind).")
    ap.add_argument("--campaign", required=True,
                    help="campaign name -> loads <campaign>.yaml (e.g. --campaign education)")
    ap.add_argument("--inputs", nargs="*", default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--all-columns", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()

    # Load the campaign config FIRST — this populates every engine global.
    try:
        load_campaign_config(args.campaign)
    except Exception as e:
        log.error("Could not load campaign config: %s", e)
        return 1

    script_dir = Path(__file__).resolve().parent
    if not args.inputs:
        log.error("No input files given. Pass --inputs <file.xlsx> ...")
        return 1
    names = args.inputs
    paths = []
    for n in names:
        p = Path(n)
        if not p.is_absolute(): p = p if p.exists() else (script_dir / n)
        if p.exists(): paths.append(p)
        else: log.error("Input not found: %s", n)
        
    if not paths:
        log.error("No valid input files. Put the .xlsx next to this script or pass --inputs.")
        return 1
        
    if not _HAVE_RAPIDFUZZ: log.warning("rapidfuzz not installed (pip install rapidfuzz).")
    if not _HAVE_TQDM: log.warning("tqdm not installed (pip install tqdm).")
    
    for p in paths:
        try:
            out = process_file(p, max_rows=args.max_rows, keep_extras=args.all_columns, use_summary=not args.no_summary)
            log.info("OUTPUT: %s", out)
        except Exception as e:
            log.exception("Failed on %s: %s", p.name, e)
            
    log.info("All done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())