import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


FORM_BY_INTENT = {
    "definition": "Answer in complete sentences that define or describe the "
                  "entity.",
    "enumeration": "Answer in prose and name EVERY element of the requested "
                   "set that the facts support – no element may be missing.",
    "verification": "Start the answer with 'Yes' or 'No', then justify it "
                    "from the facts.",
    "relation": "Describe the relationship(s) explicitly, naming the "
                "properties involved.",
    "explanation": "Explain step by step which facts lead to the conclusion.",
}

ENTITY_TOKEN = re.compile(
    r"\b(?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+|[A-Za-z]{2,}_[0-9]+)\b")

INDEX_LEAK_PATTERNS = (
    re.compile(r"\s*[\[(](?:fact|axiom|evidence)s?\s*#?\d+(?:\s*[,;]\s*#?\d+)*[\])]",
               re.I),
    re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]"),
    re.compile(r"</?think(?:ing)?>", re.I),
)


@dataclass
class GenerationResult:
    answer: str
    model_answer: str = ""
    evidence: list = field(default_factory=list)
    evidence_indices: list = field(default_factory=list)
    refused: bool = False
    abstain_reason: str = ""
    warnings: list = field(default_factory=list)
    prompt_facts: list = field(default_factory=list)
    prompt_descriptions: list = field(default_factory=list)
    error: str = ""


class AnswerGenerator:
    MODEL = "gemma2:27b"
    URL = "http://localhost:11434/v1"  # Ollama
    API_KEY = "ollama"
    TIMEOUT = 300.0  # seconds
    SEED = 42
    TEMPERATURE = 0.0
    NUM_CTX = 8192  # Ollama's default cuts off long fact lists

    MIN_ANSWER_WORDS = 4
    MAX_ATTEMPTS = 2

    ABSTAIN_NOT_IN_ONTOLOGY = ("The ontology contains no relevant knowledge to "
                               "answer this question.")
    ABSTAIN_NO_AXIOMS = ("The ontology contains no relevant knowledge to answer "
                         "this question: no matching entities or axioms were "
                         "found.")
    ABSTAIN_WITH_CONTEXT = ("The facts retrieved from the ontology do not "
                            "support an answer to this question.")
    TERMS_ABSENT = ("The term(s) {terms} do not occur anywhere in the ontology, "
                    "neither in a label or identifier nor in a description.")
    TERMS_PRESENT = ("The term(s) {terms} do occur in the ontology, but not as "
                     "an entity that carries an answer to this question.")

    def __init__(self, client=None):
        self.model = self.MODEL
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI
            self.client = OpenAI(base_url=self.URL, api_key=self.API_KEY,
                                 timeout=self.TIMEOUT)

    def generate(self, question: str, retrieval: dict, report: dict,
                 display_axioms=None, descriptions=None, unresolved=None,
                 ontology_names=None, focus=None) -> GenerationResult:
        axioms = retrieval["axioms"]
        checks = report["existence_checks"]
        unlinked = report["unlinked_terms"]

        if not axioms:
            note = self._term_note(checks, unlinked)
            return GenerationResult(answer=self._abstain_answer(note),
                                    refused=True, abstain_reason="empty_context")

        prompt_facts = self._number_facts(display_axioms, unresolved or {})
        prompt_descriptions = list(descriptions or [])
        system_prompt = self._build_system_prompt(
            prompt_facts, prompt_descriptions, retrieval,
            self._existence_block(checks, unlinked),
            focus)

        result = None
        feedback = ""
        last_error = ""
        # retry with the feedback appended, the facts stay the same
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            if feedback:
                user_content = (question + "\n\nATTENTION – your previous "
                                "output was rejected: " + feedback
                                + "\nProduce a corrected JSON object using the "
                                "SAME facts.")
            else:
                user_content = question

            try:
                raw = self._call_llm(system_prompt, user_content)
            except Exception as exc:
                last_error = "LLM-Aufruf fehlgeschlagen: " + str(exc)
                feedback = "API error: " + str(exc)
                logger.warning("   [Generator] Versuch %d: %s", attempt,
                               last_error)
                continue

            parsed, problem = self._parse_attempt(
                raw, axioms, ontology_names, unlinked,
                retrieval["enumeration_targets"])
            if parsed is None:
                last_error = problem
                feedback = problem
                logger.info("   [Generator] Versuch %d ungültig: %s", attempt,
                            problem)
                continue
            if problem and attempt < self.MAX_ATTEMPTS:
                feedback = problem
                logger.info("   [Generator] Versuch %d mangelhaft, Retry: %s",
                            attempt, problem)
                continue
            if problem:
                parsed.warnings.append("nach letztem Versuch akzeptiert trotz "
                                       "Mangel: " + problem)
            result = parsed
            break

        if result is None:
            return GenerationResult(
                answer="",
                error="Keine gültige Antwort nach " + str(self.MAX_ATTEMPTS)
                      + " Versuchen: " + last_error,
                prompt_facts=prompt_facts,
                prompt_descriptions=prompt_descriptions)

        result = self._finalize(result, axioms, ontology_names, unlinked,
                                checks)
        result.prompt_facts = prompt_facts
        result.prompt_descriptions = prompt_descriptions
        return result

    def _call_llm(self, system_prompt: str, user_content: str) -> str:
        # some models ignore the JSON format with a separate system role
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user",
                       "content": system_prompt + "\n\nQUESTION: " + user_content}],
            temperature=self.TEMPERATURE,
            seed=self.SEED,
            response_format={"type": "json_object"},
            extra_body={"options": {"num_ctx": self.NUM_CTX}},
        )
        return response.choices[0].message.content or ""

    def _build_system_prompt(self, facts: list[str], descriptions: list[str],
                             retrieval: dict, existence_block: str,
                             focus: list[str]) -> str:
        incomplete = [entry for entry in retrieval["completeness"]
                      if not entry.get("complete")]
        completeness_line = ""
        if incomplete:
            detail = "; ".join(entry["parent"] + ": " + str(entry["listed"])
                               + " of " + str(entry["total"])
                               + " subclasses listed" for entry in incomplete)
            completeness_line = ("INCOMPLETE CONTEXT (" + detail
                                 + "). Name this incompleteness in the answer "
                                 "instead of pretending the list is exhaustive.")

        sections = [
            "You answer questions STRICTLY and EXCLUSIVELY from the numbered "
            "ontology facts below. World knowledge is forbidden, even if the "
            "facts seem incomplete or wrong.",
            "",
            "FACTS:",
        ]
        sections += facts
        if descriptions:
            sections += ["", "DESCRIPTIONS (background only – NOT citable as "
                             "evidence):"]
            sections += descriptions
        if retrieval["context_summaries"]:
            sections += ["", "CONTEXT SUMMARY:"]
            sections += retrieval["context_summaries"]
        if completeness_line:
            sections += ["", completeness_line]
        if focus:
            sections += ["", "The question focuses on: " + ", ".join(focus) + "."]
        if existence_block:
            sections += ["", existence_block]

        form = FORM_BY_INTENT.get(retrieval["intent"],
                                  FORM_BY_INTENT["definition"])
        sections += [
            "",
            "FORM: " + form,
            "",
            "OUTPUT: exclusively a JSON object, no other text:",
            '{"answer": "<prose answer>", "evidence": [<indices of ALL facts '
            'used>], "abstained": <true|false>}',
            "",
            "RULES:",
            "1. Every claim in the answer must follow from the cited facts.",
            "2. evidence contains integer indices from the FACTS list only.",
            "3. If the facts cannot answer the question, set abstained=true, "
            "evidence=[] and say so in the answer.",
            "4. An answer without evidence counts as a refusal.",
            "5. Use the identifiers as written in the facts, not the "
            "labels; a label in brackets is only a display name.",
        ]
        return "\n".join(sections)

    @staticmethod
    def _number_facts(display_axioms, unresolved: dict) -> list[str]:
        lines = []
        for i, shown in enumerate(display_axioms):
            missing = unresolved.get(i)
            if missing:
                shown += ("   [NO LABEL for " + ", ".join(missing)
                          + " – identifier only, do not interpret]")
            lines.append("[" + str(i) + "] " + shown)
        return lines

    @staticmethod
    def _existence_block(existence_checks, unlinked_terms) -> str:
        if not unlinked_terms:
            return ""
        checked = {check["term"]: check for check in existence_checks}
        absent = []
        weak = []
        for term in unlinked_terms:
            check = checked.get(term)
            if check is None or check["found"]:
                weak.append(term)
            else:
                absent.append(term)

        lines = []
        if absent:
            lines.append("Terms with no counterpart in the ontology (verified "
                         "by full-text search): " + ", ".join(absent)
                         + ". Answer the question from the FACTS regardless. "
                         "Mention the absence of these terms only if the "
                         "question cannot be answered without them.")
        if weak:
            lines.append("Terms without an entity of their own in the ontology: "
                         + ", ".join(weak) + ". Do not invent entities for them. "
                         "If the question asks whether something falls under "
                         "such a term, answer from the classification given in "
                         "the FACTS.")
        return "\n".join(lines)

    def _parse_attempt(self, raw: str, axioms: list[str], ontology_names,
                       unlinked, candidates):
        try:
            payload = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            return None, "Output was not valid JSON (" + str(exc) + ")."
        if not isinstance(payload, dict):
            return None, "Output must be a JSON object."

        answer = self._sanitize_answer(str(payload.get("answer", "")))
        if not answer:
            return None, "The 'answer' field was empty."

        result = GenerationResult(answer=answer, model_answer=answer,
                                  refused=bool(payload.get("abstained")))
        indices, warnings = self._parse_indices(payload.get("evidence", []),
                                                len(axioms))
        result.warnings.extend(warnings)
        result.evidence_indices = indices
        result.evidence = [axioms[i] for i in indices]
        if result.refused:
            return result, ""

        problems = []
        if not indices:
            problems.append("You made claims but cited no evidence. Cite the "
                            "indices of ALL facts you used, or set "
                            "abstained=true.")
        if len(answer.split()) < self.MIN_ANSWER_WORDS:
            problems.append("The answer is shorter than "
                            + str(self.MIN_ANSWER_WORDS) + " words – answer in "
                            "complete sentences.")
        foreign, uncited = self._entity_issues(answer, result.evidence, axioms,
                                               ontology_names, unlinked)
        if foreign:
            problems.append("The entities " + ", ".join(foreign) + " appear in "
                            "none of the facts – use only entities from the "
                            "FACTS list.")
        if uncited:
            result.warnings.append("Entitäten ohne Deckung in den zitierten "
                                   "Axiomen: " + ", ".join(uncited))
        missing = [name for name in candidates if name not in result.answer]
        if missing:
            problems.append("The enumeration is incomplete – these elements are "
                            "missing: " + ", ".join(missing)
                            + ". Name ALL elements.")
        return result, " ".join(problems)

    def _finalize(self, result: GenerationResult, axioms: list[str],
                  ontology_names, unlinked, checks) -> GenerationResult:
        if not result.refused:
            foreign, _ = self._entity_issues(result.answer, result.evidence,
                                             axioms, ontology_names, unlinked)
            if foreign:
                # drop sentences with world knowledge
                sentences = re.split(r"(?<=[.!?])\s+", result.answer)
                kept = [sentence for sentence in sentences
                        if not any(token in sentence for token in foreign)]
                cleaned = " ".join(kept).strip()
                result.warnings.append("Sätze mit ontologiefremden Entitäten "
                                       "entfernt: " + ", ".join(foreign))
                if cleaned and len(cleaned.split()) >= self.MIN_ANSWER_WORDS:
                    result.answer = cleaned
                else:
                    result.refused = True
                    result.abstain_reason = "ungrounded_answer"

        if not result.refused and not result.evidence_indices:
            result.refused = True
            result.abstain_reason = "no_evidence_cited"
            result.warnings.append("Antwort ohne Evidenz – als Verweigerung "
                                   "gewertet.")

        if result.refused:
            logger.info("   [Generator] Verweigerung (%s), Modelltext war: %s",
                        result.abstain_reason or "model_abstained",
                        result.model_answer)
            note = self._term_note(checks, unlinked)
            result.answer = self.ABSTAIN_WITH_CONTEXT
            if note:
                result.answer += " " + note
            if not result.abstain_reason:
                result.abstain_reason = "model_abstained"
            if result.evidence_indices:
                result.warnings.append("Verweigerung zitierte Evidenz – "
                                       "Indizes verworfen: "
                                       + str(result.evidence_indices))
                result.evidence_indices = []
                result.evidence = []
        return result

    @staticmethod
    def _parse_indices(raw_indices, n_facts: int):
        if not isinstance(raw_indices, list):
            return [], ["evidence ist kein Array: " + repr(raw_indices)]
        indices = []
        warnings = []
        for value in raw_indices:
            try:
                index = int(value)
            except (TypeError, ValueError):
                warnings.append("Evidenz-Index nicht numerisch: " + repr(value))
                continue
            if 0 <= index < n_facts:
                if index not in indices:
                    indices.append(index)
            else:
                warnings.append("Evidenz-Index außerhalb des Bereichs 0.."
                                + str(n_facts - 1) + ": " + str(index))
        return indices, warnings

    @staticmethod
    def _entity_issues(answer: str, cited, context, ontology_names, unlinked):
        # terms from the question may appear in the answer
        question_tokens = set()
        for term in unlinked:
            question_tokens.update(ENTITY_TOKEN.findall(term))
        context_text = " ".join(context)
        cited_text = " ".join(cited)

        foreign = []
        uncited = []
        for token in sorted(set(ENTITY_TOKEN.findall(answer)) - question_tokens):
            known = token in context_text or (
                ontology_names is not None and token.lower() in ontology_names)
            if not known:
                foreign.append(token)
            elif token not in cited_text:
                uncited.append(token)
        return foreign, uncited

    @staticmethod
    def _sanitize_answer(text: str) -> str:
        for pattern in INDEX_LEAK_PATTERNS:
            text = pattern.sub("", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        return re.sub(r"[ \t]{2,}", " ", text).strip()

    def _abstain_answer(self, note: str) -> str:
        if not note:
            return self.ABSTAIN_NO_AXIOMS
        return self.ABSTAIN_NOT_IN_ONTOLOGY + " " + note

    def _term_note(self, existence_checks, unlinked_terms) -> str:
        absent, present = self._split_checked_terms(existence_checks,
                                                    unlinked_terms)
        parts = []
        if absent:
            parts.append(self.TERMS_ABSENT.format(terms=_quoted(absent)))
        if present:
            parts.append(self.TERMS_PRESENT.format(terms=_quoted(present)))
        return " ".join(parts)

    @staticmethod
    def _split_checked_terms(existence_checks, unlinked_terms):
        checked = {check["term"]: check for check in existence_checks}
        absent, present = [], []
        for term in unlinked_terms:
            check = checked.get(term)
            if check is None:
                continue
            if check["found"]:
                present.append(term)
            else:
                absent.append(term)
        return absent, present


def _quoted(terms) -> str:
    return ", ".join("'" + term + "'" for term in terms)
