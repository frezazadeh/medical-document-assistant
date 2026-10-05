"""Checks the citations in an answer against the sources the model was shown.

Two things can go wrong with a citation marker such as [3]:

  1. the id does not exist (the model invented it);
  2. the id exists, but that passage is not where the statement comes from.

The second one is easy to miss, because the answer looks properly cited. I saw
it in the UI: "412 participants were randomized [3]", where source 3 was the
inclusion criteria and the number was in source 4. So every cited sentence is
compared with every source, and when another source clearly supports it better
than the cited one, the marker is pointed at that source instead.

The comparison is lexical: numbers must match exactly, words are compared by
their first letters. It does not understand meaning, so it will not notice a
passage that says the opposite in the same words. It catches wrong pointers and
numbers that appear in none of the cited passages.
"""

import re
from dataclasses import dataclass, field

from app.context import NOT_FOUND_ANSWER, Source

# Matches [2] and also [1, 3], which smaller models like to write.
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# A number on its own, not a digit inside a word such as "m2" or "CYP11B2".
_NUMBER = re.compile(r"(?<![a-z\d.])\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z]{4,}")
_LIST_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
# Sentence boundaries, captured so the answer can be put back together unchanged.
_BOUNDARY = re.compile(r"((?<=[.!?])\s+(?=[A-Z0-9(])|\n+)")

_STOPWORDS = {
    "that", "this", "with", "from", "were", "have", "been", "there", "their", "which",
    "during", "about", "into", "than", "then", "also", "each", "they", "them", "what",
    "when", "will", "would", "could", "should", "according", "source", "sources",
}  # fmt: skip

# How much better another source has to match before a citation is moved to it.
REPAIR_MARGIN = 0.25
# An invented id is only replaced when some source matches the sentence at least this well.
REPAIR_FLOOR = 0.5


@dataclass(frozen=True)
class CitationCheck:
    answer: str  # the answer, with corrected markers where a citation was moved
    cited: list[Source]
    unknown_refs: list[int]
    repairs: list[dict] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    grounded: bool = False


def cited_refs(text: str) -> list[int]:
    """Source ids in the order they first appear."""
    refs: list[int] = []
    for group in _MARKER.findall(text):
        for ref in (int(n) for n in group.split(",")):
            if ref not in refs:
                refs.append(ref)
    return refs


def is_not_found(answer: str) -> bool:
    return NOT_FOUND_ANSWER.rstrip(".").lower() in answer.lower()


def check_citations(answer: str, sources: list[Source]) -> CitationCheck:
    """Resolve, and where needed correct, the citation markers in an answer.

    grounded is True when the answer is the agreed not-found sentence, or when
    it cites at least one source, cites no invented ids, and every number in a
    cited sentence appears in a passage that sentence cites.
    """
    by_ref = {s.ref: s for s in sources}
    if is_not_found(answer):
        unknown = [r for r in cited_refs(answer) if r not in by_ref]
        return CitationCheck(answer=answer, cited=[], unknown_refs=unknown, grounded=True)

    profiles = {s.ref: _profile(s.text) for s in sources}
    repairs: list[dict] = []
    unsupported: list[str] = []

    parts = _BOUNDARY.split(answer)
    for i in range(0, len(parts), 2):  # odd positions are the separators
        sentence = parts[i]
        refs = cited_refs(sentence)
        if not refs:
            continue

        numbers, words = _profile(_LIST_PREFIX.sub("", _MARKER.sub(" ", sentence)))
        valid = [r for r in refs if r in by_ref]
        better = _better_source(numbers, words, valid, profiles)
        if better is not None:
            repairs.append({"sentence": sentence.strip(), "from": refs, "to": better})
            sentence = parts[i] = _repoint(sentence, better)
            valid = [better]

        cited_numbers = set().union(*(profiles[r][0] for r in valid)) if valid else set()
        if numbers - cited_numbers:
            unsupported.append(sentence.strip())

    answer = "".join(parts)
    refs = cited_refs(answer)
    cited = [by_ref[r] for r in refs if r in by_ref]
    unknown = [r for r in refs if r not in by_ref]
    return CitationCheck(
        answer=answer,
        cited=cited,
        unknown_refs=unknown,
        repairs=repairs,
        unsupported=unsupported,
        grounded=bool(cited) and not unknown and not unsupported,
    )


def _profile(text: str) -> tuple[set[str], set[str]]:
    """The numbers in a text and its content words, cut to six letters as a crude stem."""
    lowered = text.lower()
    numbers = set(_NUMBER.findall(lowered))
    words = {w[:6] for w in _WORD.findall(lowered) if w not in _STOPWORDS}
    return numbers, words


def _support(numbers: set[str], words: set[str], profile: tuple[set[str], set[str]]) -> float:
    """0..1: how much of a sentence can be found in a source. Numbers weigh more than words."""
    source_numbers, source_words = profile
    word_hit = len(words & source_words) / len(words) if words else 0.0
    if not numbers:
        return word_hit
    number_hit = len(numbers & source_numbers) / len(numbers)
    return 0.6 * number_hit + 0.4 * word_hit


def _better_source(
    numbers: set[str],
    words: set[str],
    cited: list[int],
    profiles: dict[int, tuple[set[str], set[str]]],
) -> int | None:
    """The source a sentence should cite instead, or None if its citation can stay."""
    if not profiles:
        return None
    scores = {ref: _support(numbers, words, profile) for ref, profile in profiles.items()}
    best = max(scores, key=lambda ref: (scores[ref], -ref))
    if best in cited:
        return None

    if not cited:  # only invented ids
        return best if scores[best] >= REPAIR_FLOOR else None

    cited_score = max(scores[ref] for ref in cited)
    if scores[best] - cited_score >= REPAIR_MARGIN:
        return best

    # Not a clear win on the overall score, but the cited passages lack a number
    # that this one has in full. A number is the part that must not be wrong.
    cited_numbers = set().union(*(profiles[ref][0] for ref in cited))
    if numbers - cited_numbers and numbers <= profiles[best][0]:
        return best
    return None


def _repoint(sentence: str, ref: int) -> str:
    """Replace the markers in a sentence with a single [ref], at the first marker's position."""
    seen = False

    def swap(_match: re.Match) -> str:
        nonlocal seen
        if seen:
            return ""
        seen = True
        return f"[{ref}]"

    return _MARKER.sub(swap, sentence)
