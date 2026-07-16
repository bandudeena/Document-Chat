"""RAG accuracy evaluation against the global warming PDF (+ multi-doc corpus).

Each case has ground truth pulled from the source PDF by direct text extraction.
Grades: answer must contain required keywords (any-of groups), cite the right
file, and refusals must trigger where expected.
"""

import json
import os
import re
import urllib.request

BASE = os.environ.get("EVAL_BASE_URL", "http://localhost:8080")
GW = "global warming.pdf"

# (question, [keyword groups: each group = list of alternatives, one must match],
#  expected_source_file or None for refusal, note)
CASES = [
    (
        "What percentage does water vapor contribute to the greenhouse effect?",
        [["36"], ["72"]],
        GW,
        "table p.7: water vapor 36-72%",
    ),
    (
        "How much more heat does a methane molecule trap compared to CO2?",
        [["twenty", "20"]],
        GW,
        "p.2: methane traps >20x more heat than CO2",
    ),
    (
        "How much more effective is nitrous oxide than CO2 at trapping heat?",
        [["300"]],
        GW,
        "p.2: N2O 300x more effective",
    ),
    (
        "How much could sea levels rise by the year 2100?",
        [["20"], ["88"]],
        GW,
        "IPCC projection: 20-88 cm by 2100",
    ),
    (
        "How much has ocean acidity increased since the early 1700s?",
        [["25"]],
        GW,
        "EPA: ~25% since early 1700s",
    ),
    (
        "What temperature increase could trigger permafrost thawing in Siberia?",
        [["1.5"]],
        GW,
        "1.5°C (2.7°F) above current levels",
    ),
    (
        "What is the difference between global warming and climate change?",
        [["average temperature", "temperature"], ["rainfall", "wind", "broader"]],
        GW,
        "GW = temp rise; CC = broader changes (wind, rainfall, extremes)",
    ),
    (
        "Show me the greenhouse gas contributions as a table",
        [["| "], ["Methane", "CH"], ["Carbon dioxide", "CO"]],
        GW,
        "must render a Markdown table from the p.7 table chunk",
    ),
    (
        "What was Aurora Coffee's strongest quarter?",
        [["Q4"], ["310"]],
        "aurora-coffee-annual-report.pdf",
        "multi-doc: must retrieve from the OTHER document, not global warming",
    ),
    (
        "What is the current price of Bitcoin?",
        None,
        None,
        "not in corpus: must refuse without hallucinating",
    ),
]


def ask(question: str):
    req = urllib.request.Request(
        f"{BASE}/api/chat",
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    answer, sources, usage = "", [], {}
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            d = json.loads(line[6:])
            if d["type"] == "token":
                answer += d["text"]
            elif d["type"] == "sources":
                sources = d["sources"]
            elif d["type"] == "done":
                usage = d["usage"]
    return answer, sources, usage


def main():
    passed = 0
    total_cost = 0.0
    for question, groups, expected_file, note in CASES:
        answer, sources, usage = ask(question)
        total_cost += usage.get("estimated_cost_usd", 0)
        problems = []

        if groups is None:  # refusal expected
            refusal_markers = [
                "couldn't find", "don't have access", "no information", "not in your",
            ]
            refused = any(m in answer.lower() for m in refusal_markers)
            if not refused or sources:
                problems.append("expected refusal with no sources")
            if re.search(r"\$\s?\d", answer):
                problems.append("answer contains a fabricated price")
        else:
            for group in groups:
                if not any(k.lower() in answer.lower() for k in group):
                    problems.append(f"missing any of {group}")
            files = {s["filename"] for s in sources}
            if expected_file not in files:
                problems.append(f"expected source {expected_file}, got {files or 'none'}")
            if not re.search(r"\[\d+\]", answer) and "| " not in answer:
                problems.append("no [n] citation in answer")

        status = "PASS" if not problems else "FAIL"
        passed += status == "PASS"
        srcs = [(s["ref"], s["content_type"], s["page"], round(s["score"], 2)) for s in sources]
        print(f"[{status}] {question}")
        print(f"       truth: {note}")
        print(f"       sources: {srcs}")
        if problems:
            print(f"       PROBLEMS: {problems}")
            print(f"       answer: {answer[:400]}")
        else:
            print(f"       answer: {' '.join(answer.split())[:180]}")
        print()

    print(f"score: {passed}/{len(CASES)}   total eval cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
