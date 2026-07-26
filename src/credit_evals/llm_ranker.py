"""The LLM path: application text -> risk grade A..E + cited rationale.

Design rule carried over from econ-rag-agent: the LLM's words are
optional, its numbers are not trusted. `verify_citations()` rejects any
rationale whose figures don't appear in the application text the model
was shown; a rejected rationale falls back to a deterministic one and
the event is counted (the eval harness reports the rate).

Without ANTHROPIC_API_KEY the ranker is a deterministic scorecard stub,
so every eval runs keyless. The live path ships UNTESTED-LIVE: it has
never been run with a real key in this repo. That label comes off only
after a real keyed run, in its own commit.
"""
import os
import re
from dataclasses import dataclass

GRADES = ["A", "B", "C", "D", "E"]  # A = lowest risk
GRADE_TO_RANK = {g: i for i, g in enumerate(GRADES)}
ABSTAIN = "ABSTAIN"

_PROMPT = (
    "You are a credit risk reviewer. Grade this application's default "
    "risk as one letter A (lowest) to E (highest), then one short "
    "rationale sentence. Cite only values present in the application. "
    "If the application lacks the information to judge, reply ABSTAIN "
    "and say what is missing.\n\nFormat: GRADE: <A-E|ABSTAIN>. "
    "RATIONALE: <sentence>\n\n{application}"
)


@dataclass
class Judgment:
    grade: str          # A..E or ABSTAIN
    rationale: str
    grounded: bool      # did the rationale survive verify_citations?
    source: str         # "stub" | "live"


def verify_citations(rationale: str, application: str) -> bool:
    """Every number in the rationale must literally appear in the application.

    Same shape as econ-rag-agent's verify_numbers: a testable gate, not a
    polite instruction in the prompt. Grades A-E are exempt (they are the
    output, not evidence); everything else numeric must be grounded.
    """
    app_numbers = set(re.findall(r"\d+(?:\.\d+)?", application))
    cited = re.findall(r"\d+(?:\.\d+)?", rationale)
    return all(c in app_numbers for c in cited)


def _stub(application: str) -> tuple[str, str]:
    """Deterministic scorecard standing in for the LLM. Crude by design —
    the point of the harness is to MEASURE it, and an imperfect second
    opinion makes disagreement evals legible."""
    def field(name: str) -> str:
        m = re.search(rf"^- {name}: (.+)$", application, re.M)
        return m.group(1) if m else ""

    score = 0
    reasons = []
    status = field("status")
    if "no checking account" in status:
        score += 2; reasons.append("no checking account")
    elif "< 100 DM" in status and "0 <=" not in status:
        score += 2; reasons.append(f"checking status '{status}'")
    dur = int(field("duration") or 0)
    if dur >= 36: score += 2; reasons.append(f"duration {dur} months")
    elif dur >= 24: score += 1; reasons.append(f"duration {dur} months")
    amt = int(field("amount") or 0)
    if amt >= 8000: score += 2; reasons.append(f"amount {amt}")
    elif amt >= 4000: score += 1; reasons.append(f"amount {amt}")
    hist = field("credit_history")
    if "critical" in hist: score -= 1  # bank's own coding quirk: repeat borrowers
    if "delay in paying" in hist: score += 1; reasons.append("past payment delays")
    sav = field("savings")
    if "unknown" in sav or "< 100 DM" in sav:
        score += 1; reasons.append("thin savings")
    grade = GRADES[max(0, min(4, score))]
    rationale = ("Scorecard stub: " + "; ".join(reasons)) if reasons else \
        "Scorecard stub: no elevated-risk markers found."
    return grade, rationale


def judge(application: str) -> Judgment:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        grade, rationale = _stub(application)
        source = "stub"
    else:  # UNTESTED-LIVE: never yet run with a real key from this repo
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=120,
            messages=[{"role": "user",
                       "content": _PROMPT.format(application=application)}])
        text = msg.content[0].text
        gm = re.search(r"GRADE:\s*([A-E]|ABSTAIN)", text)
        rm = re.search(r"RATIONALE:\s*(.+)", text, re.S)
        grade = gm.group(1) if gm else ABSTAIN
        rationale = (rm.group(1).strip() if rm else text.strip())[:300]
        source = "live"

    grounded = verify_citations(rationale, application)
    if not grounded and grade != ABSTAIN:
        # Fabrication gate fired: keep the grade, replace the evidence.
        rationale = "Rationale withheld: cited figures not present in application."
    return Judgment(grade=grade, rationale=rationale,
                    grounded=grounded, source=source)
