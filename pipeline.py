import logging
import os
import time

from src.axioms import (annotate_axiom, build_label_index, describe_entities,
                        local_name, unresolved_ids)
from src.generation import AnswerGenerator
from src.linking import EntityLinker, LinkResult, known_link_names
from src.ontology import KnowledgeGraph, OntologyLoader, ReasoningEngine
from src.retrieval import GraphRetriever

logger = logging.getLogger(__name__)


class OntoQAPipeline:
    def __init__(self, ontology_file_path: str, dependencies_paths=None,
                 reasoner_name: str = ""):
        self.ontology_file = ontology_file_path
        self.dependencies = dependencies_paths or []
        self.reasoner_name = reasoner_name

        self.reasoner = None
        self.tmp = None
        self.kg = None
        self.facts = {}
        self.label_index = {}
        self.ontology_names = set()
        self.linker = None
        self.retriever = None
        self.generator = None

    def initialize(self) -> None:
        temp_path, self.tmp = OntologyLoader().load(self.ontology_file,
                                                    self.dependencies)
        self.reasoner = ReasoningEngine(temp_path, reasoner=self.reasoner_name)
        inferred_path = os.path.join(os.path.dirname(temp_path), "inferred.owl")
        self.reasoner.save_inferred(inferred_path)

        self.kg = KnowledgeGraph(inferred_path)
        self.facts = self.kg.build_ontology_facts()
        self.label_index = build_label_index(self.facts)
        self.ontology_names = self._collect_ontology_names()

        self.linker = EntityLinker(self.facts)
        self.retriever = GraphRetriever(self.kg, self.facts)
        self.generator = AnswerGenerator()

    def run_questions(self, questions: list) -> list[dict]:
        results = []
        for i, spec in enumerate(questions, start=1):
            if isinstance(spec, dict):
                question = spec["question"]
                expected = spec
            else:
                question = spec
                expected = None
            result = self.ask(question, expected=expected)
            question_id = (expected.get("id") if expected else None) \
                or "Q" + str(i).zfill(2)
            results.append({"id": question_id, **result})
        return results

    def ask(self, question: str, expected=None) -> dict:
        logger.info("Frage: %s", question)
        started = time.time()
        report = self.linker.link(question)
        retrieval = self.retriever.retrieve(report)
        generation = self._generate(question, report, retrieval)

        expected = expected or {}
        result = {
            "category": expected.get("category", ""),
            "intent": report["intent"],
            "question": question,
            "answer": generation.answer,
            "evidence": generation.evidence,
            "expected_answer": expected.get("expected_answer", ""),
            "expected_evidence": expected.get("expected_evidence", []),
            "abstain_reason": generation.abstain_reason,
            "links": [self._serialize_link(link) for link in report["links"]],
            "unlinked_terms": list(report["unlinked_terms"]),
            "existence_checks": report["existence_checks"],
            "llm_facts": generation.prompt_facts,
            "llm_descriptions": generation.prompt_descriptions,
            "warnings": generation.warnings,
            "generation_error": generation.error,
            "seconds": round(time.time() - started, 2),
        }
        logger.info("Antwort (%.1fs): %s", result["seconds"],
                    generation.answer or generation.error)
        return result

    def cleanup(self) -> None:
        if self.reasoner is not None:
            try:
                self.reasoner.shutdown()
            except Exception as exc:
                logger.warning("Reasoner-Shutdown fehlgeschlagen: %s", exc)
            self.reasoner = None
        if self.tmp is not None:
            self.tmp.cleanup()
            self.tmp = None

    def _generate(self, question: str, report: dict, retrieval: dict):
        axioms = retrieval["axioms"]
        display_axioms = [annotate_axiom(axiom, self.label_index)
                          for axiom in axioms]

        known = {local_name(iri) for iri in self.facts}
        unresolved = {}
        for i, axiom in enumerate(axioms):
            missing = unresolved_ids(axiom, self.label_index, known)
            if missing:
                unresolved[i] = missing

        seed_names = set(retrieval["seeds"])
        seed_iris = [link.iri for link in report["links"]
                     if local_name(link.iri) in seed_names]
        # without seeds, describe all linked entities
        if not seed_iris:
            seed_iris = [link.iri for link in report["links"]]
        descriptions = describe_entities(seed_iris, self.facts, self.label_index)

        focus = [annotate_axiom(name, self.label_index)
                 for name in retrieval["seeds"]]

        return self.generator.generate(
            question, retrieval, report,
            display_axioms=display_axioms,
            descriptions=descriptions,
            unresolved=unresolved,
            ontology_names=self.ontology_names,
            focus=focus)

    def _collect_ontology_names(self) -> set:
        names = set(known_link_names(self.facts))
        for iri, info in self.facts.items():
            names.add(local_name(iri).lower())
            for label in info.get("labels", []):
                names.add(label.lower())
        return names

    def _serialize_link(self, link: LinkResult) -> dict:
        return {"mention": link.mention, "entity": local_name(link.iri),
                "iri": link.iri,
                "kind": self.facts.get(link.iri, {}).get("type", ""),
                "score": round(link.score, 4), "method": link.method,
                "span": [link.start, link.end], "ic": link.ic}
