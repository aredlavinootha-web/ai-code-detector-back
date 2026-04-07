"""
tier2.py — Statistical / Token-level Analysis (Backend)

Upgraded Tier 2 using:
  - Word frequency database (harshnative/words-dataset, 610K words) for log-rank scoring
  - NLTK for n-gram entropy with proper tokenization and normalization
  - taaled for MATTR (Moving-Average Type-Token Ratio)
  - Language-agnostic + language-specific Bayesian feature detection
  - Smart code tokenizer that splits camelCase, snake_case, SCREAMING_CASE
"""

import math
import re
import csv
import os
import logging
from collections import Counter

import nltk
from nltk.util import ngrams as nltk_ngrams
from taaled import ld

log = logging.getLogger("tier2")

_word_ranks: dict[str, int] = {}
_resources_loaded = False

WEIGHTS = {
    "ngram_entropy":     0.10,   # inverted: low entropy → AI
    "log_rank":          0.30,
    "token_histogram":   0.20,
    "mattr":             0.10,   # inverted: low MATTR → AI
    "bayesian_features": 0.30,
}


# ─────────────────────────────────────────────────────────────────────────────
# Resource loading
# ─────────────────────────────────────────────────────────────────────────────

def load_resources():
    """Load word frequency data and NLTK tokenizer at startup."""
    global _word_ranks, _resources_loaded

    nltk.download("punkt_tab", quiet=True)

    csv_path = os.path.join(os.path.dirname(__file__), "data", "wordsFreq.csv")
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header: index,word,frequency
            for rank, row in enumerate(reader, 1):
                if len(row) >= 3:
                    word = row[1].strip().lower()
                    if word:
                        _word_ranks[word] = rank
        log.info("Loaded %d word frequencies", len(_word_ranks))
    else:
        log.warning("Word frequency file not found: %s", csv_path)

    _resources_loaded = True


def resources_ready() -> bool:
    return _resources_loaded


# ─────────────────────────────────────────────────────────────────────────────
# Smart code tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_code(code: str) -> list[str]:
    """
    Split source code into component words.
    Handles camelCase, PascalCase, snake_case, SCREAMING_CASE.
    Filters out single-character fragments and language keywords.
    """
    identifiers = re.findall(r"[a-zA-Z_]\w*", code)
    words = []
    for ident in identifiers:
        # camelCase / PascalCase: getUserById → get_User_By_Id
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", ident)
        # consecutive capitals: HTMLParser → HTML_Parser
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
        for part in s.split("_"):
            if part and len(part) > 1:
                words.append(part.lower())
    return words


# ─────────────────────────────────────────────────────────────────────────────
# 1. Log-Rank Scoring (word-frequency-based)
# ─────────────────────────────────────────────────────────────────────────────

AI_VOCABULARY = frozenset([
    "calculate", "compute", "process", "retrieve", "generate", "initialize",
    "validate", "implement", "execute", "perform", "determine", "establish",
    "utilise", "utilize", "ensure", "provide", "represent", "maintain",
    "indicate", "demonstrate", "facilitate", "leverage", "streamline",
    "optimize", "orchestrate", "instantiate", "encapsulate", "traverse",
    "iterate", "aggregate", "decouple", "designate", "configure", "invoke",
    "serialize", "deserialize", "authenticate", "authorize", "sanitize",
    "normalize", "interpolate", "propagate", "allocate", "deallocate",
    "dispatch", "intercept", "transform", "evaluate", "concatenate",
    "enumerate", "specify", "construct", "reconstruct", "comprehensive",
    "appropriate", "efficient", "effective", "robust", "seamless", "modular",
    "scalable", "maintainable", "reusable", "extensible", "immutable",
    "asynchronous", "concurrent", "idempotent", "deterministic", "polymorphic",
    "declarative", "imperative", "functionality", "implementation",
    "methodology", "paradigm", "abstraction", "parameters", "arguments",
    "attributes", "middleware", "endpoint", "payload", "schema", "serializer",
    "validator", "repository", "singleton", "factory", "observer", "decorator",
    "adapter", "configuration", "dependency", "initializer", "constructor",
    "destructor", "responsible", "specified", "successfully", "correctly",
    "properly", "accordingly", "additionally", "furthermore", "therefore",
    "consequently", "subsequently", "respectively", "moreover", "nevertheless",
    "specifically", "essentially", "fundamentally", "neighboring",
    "representing", "destination", "identifier", "structure", "structures",
])

HUMAN_VOCABULARY = frozenset([
    "tmp", "buf", "ret", "val", "idx", "ptr", "cnt", "cfg", "msg", "err",
    "ok", "num", "src", "dst", "ctx", "req", "res", "cb", "fn", "str",
    "len", "pos", "cur", "prev", "fmt", "pkg", "desc", "max", "min", "avg",
    "acc", "sum", "hack", "quick", "temp", "fixme", "todo", "wtf", "idk",
    "check", "maybe", "rough", "kludge", "nah", "hmm", "ugh", "wip", "nope",
    "yep", "aka", "misc", "junk", "crap", "oops", "meh", "stub", "mock",
    "dummy", "workaround", "bandaid", "monkey", "patch", "sketchy", "janky",
    "brittle", "flaky", "dodgy", "old", "bad", "good", "test", "debug",
    "fix", "try", "run", "get", "set", "add", "del", "pop", "push", "swap",
    "skip", "done", "fail", "pass", "drop", "grab", "tweak", "impl", "prob",
    "buggy", "nvm", "asap", "btw", "fyi", "imo", "lol",
])


def log_rank_score(code: str) -> int:
    """
    Combines three signals using the 610K word-frequency database:

    1. Curated word-list matching — the smart tokenizer splits compound
       identifiers so 'calculateOptimalRoute' now matches 'calculate' and
       'optimal' in the AI vocabulary set.
    2. Dictionary coverage — AI code uses real English words; human code
       uses abbreviations (tmp, cfg, buf) not in the dictionary.
    3. Formality — AI favours moderately rare but valid English words
       (rank 5K-100K); humans stick to extremely common ones or jargon.

    Returns 0-100, high = AI-like.
    """
    words = _tokenize_code(code)
    if len(words) < 10:
        return 50

    ai_hits = sum(1 for w in words if w in AI_VOCABULARY)
    human_hits = sum(1 for w in words if w in HUMAN_VOCABULARY)

    # Word-list net rate (primary signal)
    net_rate = (ai_hits - human_hits) / len(words)
    list_score = max(0, min(100, round(50 + net_rate * 600)))

    if not _word_ranks:
        return list_score

    # Dictionary coverage: AI code = mostly known words
    unknown_count = sum(1 for w in words if w not in _word_ranks)
    unknown_ratio = unknown_count / len(words)
    coverage_score = max(0, min(100, round(90 - unknown_ratio * 160)))

    # Formality: words in rank 5K-100K are "educated but not everyday"
    formal_count = sum(
        1 for w in words
        if w in _word_ranks and 5_000 < _word_ranks[w] < 100_000
    )
    formal_ratio = formal_count / len(words)
    formal_score = max(0, min(100, round(40 + formal_ratio * 400)))

    return round(list_score * 0.50 + coverage_score * 0.25 + formal_score * 0.25)


# ─────────────────────────────────────────────────────────────────────────────
# 2. N-gram Entropy (NLTK)
# ─────────────────────────────────────────────────────────────────────────────

def ngram_entropy(code: str, n: int = 2) -> int:
    """
    Bigram entropy using NLTK word_tokenize, filtered to alphabetic tokens
    only (drops punctuation / operators that add noise for code).

    Normalized by log2(unique_ngram_count) — the correct theoretical maximum.

    LOW entropy = predictable / repetitive = AI-like.
    Returns 0-100, where 100 = maximum diversity.
    """
    try:
        raw_tokens = nltk.word_tokenize(code.lower())
    except Exception:
        raw_tokens = code.lower().split()

    tokens = [t for t in raw_tokens if t[0].isalpha()]

    if len(tokens) < n + 2:
        return 50

    grams = list(nltk_ngrams(tokens, n))
    freq = Counter(grams)
    total = sum(freq.values())

    if total == 0:
        return 50

    H = -sum((c / total) * math.log2(c / total) for c in freq.values())
    num_unique = len(freq)
    max_H = math.log2(num_unique) if num_unique > 1 else 1

    normalized = (H / max_H) * 100 if max_H > 0 else 50
    return min(100, round(normalized))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Token Frequency Histogram (GLTR-inspired)
# ─────────────────────────────────────────────────────────────────────────────

def token_histogram_score(code: str) -> int:
    """
    Measures how concentrated token usage is.
    Uses smart tokenizer to split compound identifiers.

    HIGH concentration = narrow vocabulary = AI-like.
    Returns 0-100, high = AI-like.
    """
    words = _tokenize_code(code)
    if len(words) < 15:
        return 40

    freq = Counter(words)
    sorted_types = freq.most_common()
    total = len(words)

    top_count = max(1, round(len(sorted_types) * 0.20))
    top_tokens = set(w for w, _ in sorted_types[:top_count])
    top_usage = sum(1 for w in words if w in top_tokens)
    concentration = top_usage / total

    if concentration > 0.75:
        return 85
    if concentration > 0.65:
        return 70
    if concentration > 0.55:
        return 55
    if concentration > 0.45:
        return 40
    return 25


# ─────────────────────────────────────────────────────────────────────────────
# 4. MATTR (taaled)
# ─────────────────────────────────────────────────────────────────────────────

def mattr_score(code: str, window_size: int = 50) -> int:
    """
    Moving-Average Type-Token Ratio via the taaled package.

    LOW MATTR = narrow per-window vocabulary = AI-like.
    Returns 0-100, where 100 = high diversity (human-like).
    """
    words = _tokenize_code(code)

    if len(words) < 10:
        return 50

    if len(words) < window_size:
        unique = len(set(words))
        return min(100, round((unique / len(words)) * 130))

    try:
        ldvals = ld.lexdiv(words)
        mattr_val = ldvals.mattr
        return min(100, round(mattr_val * 130))
    except Exception as e:
        log.warning("taaled MATTR failed, using manual fallback: %s", e)
        ttrs = []
        for i in range(len(words) - window_size + 1):
            window = words[i : i + window_size]
            ttrs.append(len(set(window)) / window_size)
        mean_ttr = sum(ttrs) / len(ttrs)
        return min(100, round(mean_ttr * 130))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Bayesian Feature Score (language-agnostic + language-specific)
# ─────────────────────────────────────────────────────────────────────────────

def bayesian_feature_score(code: str, language: str = "Python") -> int:
    """
    Weighted boolean feature tests.
    Universal signals that work across all languages, plus per-language extras.

    Returns 0-100, high = AI-like.
    """
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]
    lang = language.lower()
    score = 50

    # ── Universal AI signals ──────────────────────────────────────────────

    # Formal / academic vocabulary density
    formal_hits = len(re.findall(
        r"\b(ensure|validate|initialize|implement|leverage|orchestrate|"
        r"utilize|facilitate|comprehensive|robust|seamless|furthermore|"
        r"subsequently|accordingly|consequently|appropriate|efficient|"
        r"modular|scalable|maintainable|extensible|deterministic)\b",
        code, re.I,
    ))
    if formal_hits >= 5:
        score += 10
    elif formal_hits >= 2:
        score += 6

    # Step-numbered comments (very strong AI signal)
    step_comments = len(re.findall(
        r"^\s*(?:#|//|/\*)\s*(?:Step|Phase)\s+\d+", code, re.M
    ))
    if step_comments >= 3:
        score += 10
    elif step_comments >= 2:
        score += 7

    # Verbose error messages (> 40 chars)
    err_msgs = re.findall(
        r"(?:raise|throw|Error|Exception|panic)\s*\(\s*[\"'`]([^\"'`]+)[\"'`]", code
    )
    verbose_errs = sum(1 for m in err_msgs if len(m) > 40)
    if verbose_errs >= 2:
        score += 7
    elif verbose_errs >= 1:
        score += 4

    # Trailing commas in multi-line structures
    trailing_commas = len(re.findall(r",\s*\n\s*[}\])]", code))
    if trailing_commas >= 5:
        score += 6
    elif trailing_commas >= 3:
        score += 4

    # Documentation coverage: every function has a docstring/comment above it
    func_defs = len(re.findall(
        r"(?:def |function |func |fn |"
        r"(?:public|private|protected)\s+(?:static\s+)?\w+\s+\w+\s*\()",
        code,
    ))
    doc_before_func = len(re.findall(
        r'(?:"""|\'\'\'|/\*\*|///)\s*\n\s*'
        r"(?:def |function |func |fn |"
        r"(?:public|private|protected))",
        code,
    ))
    if func_defs >= 3 and doc_before_func >= func_defs * 0.8:
        score += 8

    # Uniform function lengths (low coefficient of variation)
    func_starts = [
        i for i, l in enumerate(lines)
        if re.match(r"\s*(?:def |function |func |fn )", l)
    ]
    if len(func_starts) >= 3:
        lengths = []
        for j in range(len(func_starts)):
            end = func_starts[j + 1] if j + 1 < len(func_starts) else len(lines)
            lengths.append(end - func_starts[j])
        mean_len = sum(lengths) / len(lengths)
        if mean_len > 0:
            variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
            cv = (variance**0.5) / mean_len
            if cv < 0.3:
                score += 6

    # Zero dead / commented-out code
    commented_code_lines = re.findall(
        r"^\s*(?:#|//)\s*(?:def |class |function |return |if |for |while )", code, re.M
    )
    if len(commented_code_lines) == 0 and len(non_empty) > 20:
        score += 4

    # ── Universal human signals ───────────────────────────────────────────

    # TODO / FIXME / HACK markers
    todo_markers = len(re.findall(r"\b(TODO|FIXME|HACK|WIP|XXX|TEMP|KLUDGE)\b", code))
    if todo_markers >= 3:
        score -= 10
    elif todo_markers >= 1:
        score -= 7

    # Abbreviated variable names
    abbrev = len(re.findall(
        r"\b(tmp|buf|ret|val|idx|ptr|cnt|cfg|msg|err|ctx|req|res|cb|fn|"
        r"str|len|pos|cur|prev|fmt|pkg|acc|num|src|dst)\b",
        code,
    ))
    if abbrev >= 5:
        score -= 8
    elif abbrev >= 2:
        score -= 5

    # Debug prints left in
    debug_prints = len(re.findall(
        r"(?:console\.log|print\s*\(|fmt\.Print|println!|"
        r"System\.out\.print|printf\s*\(|log\.debug)\s*\(",
        code,
    ))
    if debug_prints >= 3:
        score -= 8
    elif debug_prints >= 1:
        score -= 4

    # Commented-out code
    if len(commented_code_lines) >= 3:
        score -= 8
    elif len(commented_code_lines) >= 1:
        score -= 5

    # Bare / empty catch blocks
    bare_catch = len(re.findall(
        r"(?:except\s*:|catch\s*\(\s*\w*\s*\)\s*\{\s*\})", code
    ))
    if bare_catch >= 1:
        score -= 6

    # Short casual error messages (< 15 chars)
    short_errs = sum(1 for m in err_msgs if len(m) < 15)
    if short_errs >= 2:
        score -= 6
    elif short_errs >= 1:
        score -= 4

    # Informal language in comments
    informal = len(re.findall(
        r"(?:#|//)\s*.*\b(wtf|idk|hmm|ugh|nah|crap|oops|meh|lol|smh|"
        r"nope|yep|dunno|gonna|wanna|kinda|sorta)\b",
        code, re.I,
    ))
    if informal >= 1:
        score -= 7

    # Mixed quote styles (inconsistency → human)
    singles = code.count("'")
    doubles = code.count('"')
    if singles > 3 and doubles > 3:
        ratio = min(singles, doubles) / max(singles, doubles)
        if ratio > 0.3:
            score -= 5

    # Magic numbers
    magic = len(re.findall(
        r"(?<!=\s)\b(?:[2-9]\d{1,4}|[1-9]\d{4,})\b(?!\s*[=;])", code
    ))
    if magic >= 5:
        score -= 6
    elif magic >= 2:
        score -= 3

    # ── Language-specific signals ─────────────────────────────────────────

    if lang == "python":
        if re.search(r'"""[\s\S]{5,200}"""', code):
            score += 6
        if len(re.findall(r'"""[\s\S]*?"""', code)) > 3:
            score += 7
        if re.search(r"from typing import", code):
            score += 6
        if re.search(r"\b(Parameters|Returns|Raises|Attributes)\s*:", code):
            score += 8
        if re.search(r"except\s+Exception\s+as\b", code):
            score += 5
        if re.search(r"^\s*except\s*:", code, re.M):
            score -= 6

    elif lang in ("javascript", "typescript"):
        jsdoc_params = len(re.findall(r"^\s*\*\s*@param\b", code, re.M))
        if jsdoc_params >= 3:
            score += 8
        elif jsdoc_params >= 1:
            score += 5
        if re.search(r"^\s*\*\s*@returns?\b", code, re.M):
            score += 5
        if re.search(r"^\s*\*\s*@throws?\b", code, re.M):
            score += 4
        if re.search(r"eslint-disable", code):
            score -= 5
        if lang == "typescript" and re.search(r"\bany\b", code):
            score -= 4

    elif lang == "java":
        javadoc_blocks = len(re.findall(r"/\*\*[\s\S]*?\*/", code))
        if javadoc_blocks >= 3:
            score += 7
        if len(re.findall(r"@Override", code)) >= 3:
            score += 5
        if re.search(r"@SuppressWarnings", code):
            score -= 5
        if re.search(r"System\.out\.println", code):
            score -= 4

    elif lang == "go":
        go_doc = len(re.findall(r"^//\s+\w+\s+", code, re.M))
        if go_doc >= 3:
            score += 6
        err_checks = len(re.findall(r"if\s+err\s*!=\s*nil", code))
        if err_checks >= 5:
            score += 5
        if re.search(r"fmt\.Println", code):
            score -= 4

    elif lang == "rust":
        rust_doc = len(re.findall(r"^\s*///\s+", code, re.M))
        if rust_doc >= 5:
            score += 7
        unwraps = len(re.findall(r"\.unwrap\(\)", code))
        if unwraps >= 3:
            score -= 6
        if re.search(r"#\[allow\(dead_code\)\]", code):
            score -= 4

    elif lang.lower() in ("c++", "cpp"):
        if re.search(r"using namespace std", code):
            score -= 4
        smart_ptrs = len(re.findall(
            r"(?:unique_ptr|shared_ptr|make_unique|make_shared)", code
        ))
        if smart_ptrs >= 3:
            score += 5
        if re.search(r"\bprintf\s*\(", code):
            score -= 4

    return max(0, min(100, round(score)))


# ─────────────────────────────────────────────────────────────────────────────
# Run all + score
# ─────────────────────────────────────────────────────────────────────────────

def run_tier2(code: str, language: str = "Python") -> dict:
    """Run all five Tier 2 analyses and return metric dict."""
    return {
        "ngram_entropy":     ngram_entropy(code),
        "log_rank":          log_rank_score(code),
        "token_histogram":   token_histogram_score(code),
        "mattr":             mattr_score(code),
        "bayesian_features": bayesian_feature_score(code, language),
    }


def score_tier2(metrics: dict) -> int:
    """Weighted combination of Tier 2 metrics. 0-100, high = AI-like."""
    return round(
        (100 - metrics["ngram_entropy"]) * WEIGHTS["ngram_entropy"]
        + metrics["log_rank"]            * WEIGHTS["log_rank"]
        + metrics["token_histogram"]     * WEIGHTS["token_histogram"]
        + (100 - metrics["mattr"])       * WEIGHTS["mattr"]
        + metrics["bayesian_features"]   * WEIGHTS["bayesian_features"]
    )
