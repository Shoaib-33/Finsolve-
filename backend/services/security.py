import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


RATE_LIMIT_PER_MINUTE = 20
DAILY_TOKEN_BUDGET = 100_000
MAX_INPUT_CHARS = 4_000

_rate_windows: dict[str, deque[float]] = defaultdict(deque)
_daily_usage: dict[tuple[str, str], int] = defaultdict(int)


PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone": r"\b(\+?\d{1,3}[\s.-])?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
    "nid": r"\b\d{13,17}\b",
    "passport": r"\b[A-Z]{2}\d{7}\b",
    "credit_card": r"\b(?:\d[ -]?){15,16}\b",
    "ip_address": r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
    "dob": r"\b(0?[1-9]|[12]\d|3[01])[\/\-](0?[1-9]|1[0-2])[\/\-](\d{2}|\d{4})\b",
}

PROMPT_INJECTION_PATTERNS = [
    r"\bignore (all )?(previous|prior|above) instructions\b",
    r"\bdisregard (all )?(previous|prior|above) instructions\b",
    r"\breveal (the )?(system|developer) prompt\b",
    r"\bshow (the )?(system|developer) prompt\b",
    r"\bact as\b.*\bjailbreak\b",
    r"\bdeveloper mode\b",
    r"\bdo anything now\b",
    r"\bDAN mode\b",
    r"\boverride (the )?(system|safety|security) rules\b",
]

TOXIC_OR_HARMFUL_PATTERNS = [
    r"\bhow to hack\b",
    r"\bhow to exploit\b",
    r"\bsql injection\b",
    r"\bbypass security\b",
    r"\bsteal credentials\b",
    r"\bcredential theft\b",
    r"\bmalware\b",
    r"\bransomware\b",
]


@dataclass
class SecurityResult:
    query: str
    blocked: bool = False
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    estimated_tokens: int = 0


def run_request_security_pipeline(query: str, user_key: str) -> SecurityResult:
    normalized = normalize_query(query)
    enforce_rate_limit(user_key)

    if len(normalized) > MAX_INPUT_CHARS:
        normalized = normalized[:MAX_INPUT_CHARS]
        warnings = [f"Input was truncated to {MAX_INPUT_CHARS} characters."]
    else:
        warnings = []

    estimated_tokens = estimate_tokens(normalized)
    enforce_daily_token_budget(user_key, estimated_tokens)

    blocked, reason = scan_prompt_injection(normalized)
    if blocked:
        return SecurityResult(query=normalized, blocked=True, reason=reason, warnings=warnings, estimated_tokens=estimated_tokens)

    blocked, reason = scan_harmful_content(normalized)
    if blocked:
        return SecurityResult(query=normalized, blocked=True, reason=reason, warnings=warnings, estimated_tokens=estimated_tokens)

    masked_query, pii_types = mask_pii(normalized)
    if pii_types:
        warnings.append(f"Sensitive input was masked: {', '.join(sorted(pii_types))}.")

    return SecurityResult(query=masked_query, warnings=warnings, estimated_tokens=estimated_tokens)


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def enforce_rate_limit(user_key: str) -> None:
    now = time.time()
    window = _rate_windows[user_key]
    while window and now - window[0] > 60:
        window.popleft()

    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise PermissionError("Rate limit exceeded. Please wait a moment before sending another request.")

    window.append(now)


def enforce_daily_token_budget(user_key: str, estimated_tokens: int) -> None:
    day = time.strftime("%Y-%m-%d", time.gmtime())
    key = (user_key, day)
    if _daily_usage[key] + estimated_tokens > DAILY_TOKEN_BUDGET:
        raise PermissionError("Daily token budget exceeded for this user.")
    _daily_usage[key] += estimated_tokens


def scan_prompt_injection(query: str) -> tuple[bool, str]:
    query_lower = query.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            return True, "Prompt-injection attempt blocked."
    return False, ""


def scan_harmful_content(query: str) -> tuple[bool, str]:
    query_lower = query.lower()
    for pattern in TOXIC_OR_HARMFUL_PATTERNS:
        if re.search(pattern, query_lower):
            return True, "This type of request is not permitted."
    return False, ""


def mask_pii(text: str) -> tuple[str, set[str]]:
    found = set()
    masked = text
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, masked):
            found.add(pii_type)
            masked = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", masked)
    return masked, found
