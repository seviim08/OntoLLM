import csv
import json
import logging
import os
import re
from statistics import mean, median

from src.stopwords import STOPWORDS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FIELDNAMES = ["id", "f1_antwort", "f1_evidenz"]

RESULT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT_NAME_SYSTEM_PIZZA = "results/pizza/result_pizza_2026-08-26T12-06-26.json"
RESULT_NAME_SYSTEM_DOID = "results/doid/result_doid_2026-08-26T12-08-43.json"

RESULT_NAME_GEMINI_DOID = "results_gemini_doid.json"
RESULT_NAME_GEMINI_PIZZA = "results_gemini_pizza.json"

RUNS = (
    ("System Pizza", RESULT_NAME_SYSTEM_PIZZA),
    ("System DOID", RESULT_NAME_SYSTEM_DOID),
    ("Gemini Pizza", RESULT_NAME_GEMINI_PIZZA),
    ("Gemini DOID", RESULT_NAME_GEMINI_DOID),
)


def main() -> None:
    runs = [evaluate_run(name, path) for name, path in RUNS]

    json_path = os.path.join(RESULT_DIR, "evaluation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"runs": runs}, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(RESULT_DIR, "evaluation_metrics.csv")
    write_csv(runs, csv_path)

    logger.info("Gespeichert: %s / %s", json_path, csv_path)
    for run in runs:
        logger.info("%-14s %s", run["name"],
                    json.dumps(run["aggregate"], ensure_ascii=False))


def evaluate_run(name: str, result_name: str) -> dict:
    with open(os.path.join(RESULT_DIR, result_name), encoding="utf-8") as f:
        payload = json.load(f)
    results = payload.get("results", payload)
    evaluation = evaluate(results)
    return {"name": name, "file": result_name, **evaluation}


def evaluate(results: list) -> dict:
    rows = []
    for i, result in enumerate(results):
        evidence = result.get("evidence", [])
        expected_evidence = result.get("expected_evidence", [])
        # a wrong abstention fails both scores
        if verweigerung_verfehlt(evidence, expected_evidence):
            answer_f1 = evidence_f1 = 0.0
        else:
            answer_f1 = f1_antwort(result.get("answer", ""),
                                   result.get("expected_answer", ""))
            evidence_f1 = f1_evidenz(evidence, expected_evidence)
        rows.append({
            "id": result.get("id", f"Q{i + 1:02d}"),
            "f1_antwort": round(answer_f1, 4),
            "f1_evidenz": round(evidence_f1, 4),
        })
    return {"per_question": rows, "aggregate": aggregate(rows)}


def verweigerung_verfehlt(evidence, expected_evidence) -> bool:
    return bool(axiom_set(evidence)) != bool(axiom_set(expected_evidence))


def f1_antwort(answer: str, reference: str) -> float:
    return f1(token_set(answer), token_set(reference))


def f1_evidenz(evidence, reference) -> float:
    return f1(axiom_set(evidence), axiom_set(reference))


def f1(produced: set, expected: set) -> float:
    # both empty means a correct abstention
    if not produced or not expected:
        return 1.0 if produced == expected else 0.0
    overlap = len(produced & expected)
    if not overlap:
        return 0.0
    precision = overlap / len(produced)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def token_set(text: str) -> set:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return {token for token in text.split() if token not in STOPWORDS}


def axiom_set(axioms) -> set:
    # no lowercasing, entity names are case sensitive
    return {" ".join(str(axiom).split()) for axiom in axioms
            if str(axiom).strip()}


def aggregate(rows: list) -> dict:
    return {
        "questions": len(rows),
        "mean_f1_antwort": round(mean(r["f1_antwort"] for r in rows), 4),
        "mean_f1_evidenz": round(mean(r["f1_evidenz"] for r in rows), 4),
        "median_f1_antwort": round(median(r["f1_antwort"] for r in rows), 4),
        "median_f1_evidenz": round(median(r["f1_evidenz"] for r in rows), 4),
    }


def write_csv(runs: list, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for i, run in enumerate(runs):
            if i:
                writer.writerow([])
            writer.writerow([run["name"]])
            writer.writerow(FIELDNAMES)
            for row in run["per_question"]:
                writer.writerow([row[key] for key in FIELDNAMES])
            summary = run["aggregate"]
            writer.writerow(["MEAN", summary["mean_f1_antwort"],
                             summary["mean_f1_evidenz"]])
            writer.writerow(["MEDIAN", summary["median_f1_antwort"],
                             summary["median_f1_evidenz"]])


if __name__ == "__main__":
    main()
