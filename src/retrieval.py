import logging
import re
from dataclasses import dataclass

from .axioms import (KIND_ASSERTION, KIND_EQUIVALENCE, KIND_PROPERTY,
                     KIND_RESTRICTION, KIND_TAXONOMY, axiom_kind, axiom_tokens,
                     local_name, unique)
from .linking import (INTENT_DEFINITION, INTENT_ENUMERATION,
                      INTENT_EXPLANATION, INTENT_RELATION,
                      INTENT_VERIFICATION, LinkResult)
from .ontology import CLASS, PROPERTIES, KnowledgeGraph

logger = logging.getLogger(__name__)


TYPE_WEIGHTS = {
    INTENT_DEFINITION: {KIND_EQUIVALENCE: 1.15, KIND_RESTRICTION: 1.1,
                        KIND_TAXONOMY: 1.0, KIND_PROPERTY: 0.9,
                        KIND_ASSERTION: 0.9},
    INTENT_ENUMERATION: {KIND_TAXONOMY: 1.2, KIND_ASSERTION: 1.1,
                         KIND_EQUIVALENCE: 1.0, KIND_RESTRICTION: 0.9,
                         KIND_PROPERTY: 0.8},
    INTENT_VERIFICATION: {KIND_PROPERTY: 1.15, KIND_RESTRICTION: 1.1,
                          KIND_EQUIVALENCE: 1.05, KIND_TAXONOMY: 1.0,
                          KIND_ASSERTION: 1.0},
    INTENT_RELATION: {KIND_PROPERTY: 1.2, KIND_RESTRICTION: 1.15,
                      KIND_ASSERTION: 1.05, KIND_TAXONOMY: 0.9,
                      KIND_EQUIVALENCE: 0.9},
    INTENT_EXPLANATION: {KIND_RESTRICTION: 1.15, KIND_EQUIVALENCE: 1.15,
                         KIND_TAXONOMY: 1.05, KIND_PROPERTY: 0.95,
                         KIND_ASSERTION: 0.95},
}

KIND_PRIORITY = {KIND_EQUIVALENCE: 0, KIND_RESTRICTION: 1, KIND_TAXONOMY: 2,
                 KIND_PROPERTY: 3, KIND_ASSERTION: 4}

SUBCLASS_AXIOM = re.compile(r"^SubClassOf\((\w+), (\w+)\)$")


@dataclass
class AxiomRecord:
    axiom: str
    score: float
    rule: str
    hops: int
    source_entity: str
    kind: str = ""
    protected: bool = False
    reserved: bool = False


class GraphRetriever:
    MIN_AXIOM_SCORE = 0.08
    MIN_PER_KIND = 1
    MAX_SUBCLASSES = 50
    BACKREF_MIN_SPECIFICITY = 0.15  # no back references for very general seeds

    RULE_WEIGHTS = {"direkt": 1.0, "property": 0.95,
                    "rückverweis": 0.9, "subklassen": 0.8}

    def __init__(self, kg: KnowledgeGraph, facts: dict[str, dict]):
        self.facts = facts
        self.specificity = KnowledgeGraph.specificity_index(facts)
        self.filler_users = kg.restriction_filler_index()
        self.property_frames = self._build_property_frames()
        self.iri_by_local = {local_name(iri): iri for iri in facts}

    def retrieve(self, report: dict) -> dict:
        intent = report["intent"]
        coverage = report["coverage_question"]
        if not report["links"]:
            return self._empty_result(intent)

        seeds = report["seeds"]
        linked = self._unique_links(report["links"])
        linked_iris = {link.iri for link in linked}
        linked_names = {local_name(iri) for iri in linked_iris}

        records = []
        records += self._rule_direct(linked, intent)
        records += self._rule_subclasses(seeds, intent, coverage)
        records += self._rule_property_frame(linked, intent)
        records += self._rule_backreference(seeds, linked_iris, intent)

        merged = self._merge(records)
        # every axiom has to mention a linked entity
        grounded = self._only_linked(merged, linked_names)
        selected = self._select(grounded, linked_names)

        listed = self._subclass_counts(selected)
        if intent == INTENT_ENUMERATION or coverage:
            enumeration_parents = {local_name(l.iri) for l in seeds}
        else:
            enumeration_parents = set()
        if intent == INTENT_ENUMERATION:
            targets = self._enumeration_targets(seeds, selected)
        else:
            targets = []

        summaries = self._subclass_summaries(listed)
        if targets:
            summaries.append("enumeration_candidates = "
                             + ", ".join(sorted(targets)))

        self._log_selection(selected)
        return {
            "axioms": [record.axiom for record in selected],
            "intent": intent,
            "context_summaries": summaries,
            "seeds": [local_name(l.iri) for l in seeds],
            "completeness": self._completeness(listed, enumeration_parents),
            "enumeration_targets": targets,
        }

    def _rule_direct(self, links, intent: str) -> list[AxiomRecord]:
        records = []
        for link in links:
            for axiom in self._profile(link.iri):
                protected = axiom_kind(axiom) in (KIND_EQUIVALENCE,
                                                  KIND_RESTRICTION)
                records.append(self._record(axiom, link, "direkt", intent,
                                            hops=0, source_iri=link.iri,
                                            protected=protected))
        return records

    def _rule_subclasses(self, seeds, intent: str,
                         coverage: bool) -> list[AxiomRecord]:
        enumerating = intent == INTENT_ENUMERATION or coverage
        records = []
        for link in seeds:
            info = self.facts.get(link.iri)
            if not info or info.get("type") != CLASS:
                continue

            parent = local_name(link.iri)
            children = self._children(link.iri)
            # very broad classes would flood the prompt
            if len(children) > self.MAX_SUBCLASSES:
                logger.info("   [Retrieval] '%s' hat %d direkte Unterklassen – "
                            "nur die ersten %d im Kontext.", parent,
                            len(children), self.MAX_SUBCLASSES)
                children = children[:self.MAX_SUBCLASSES]

            for child in children:
                axiom = ("SubClassOf(" + local_name(child) + ", " + parent + ")")
                records.append(self._record(
                    axiom, link, "subklassen", intent, hops=1,
                    source_iri=link.iri, reserved=enumerating))
            if enumerating:
                for instance in info.get("instances", []):
                    axiom = ("ClassAssertion(" + parent + ", "
                             + local_name(instance["iri"]) + ")")
                    records.append(self._record(
                        axiom, link, "subklassen", intent, hops=1,
                        source_iri=link.iri, reserved=True))
        return records

    def _rule_property_frame(self, linked, intent: str) -> list[AxiomRecord]:
        records = []
        seen = set()
        for link in linked:
            for prop_iri in self.property_frames.get(link.iri, []):
                if prop_iri in seen:
                    continue
                seen.add(prop_iri)
                for axiom in self._profile(prop_iri):
                    if axiom_kind(axiom) == KIND_PROPERTY:
                        records.append(self._record(axiom, link, "property",
                                                    intent, hops=1,
                                                    source_iri=prop_iri))
        return records

    def _rule_backreference(self, seeds, linked_iris: set,
                            intent: str) -> list[AxiomRecord]:
        records = []
        for seed in seeds:
            if self.specificity.get(seed.iri, 1.0) < self.BACKREF_MIN_SPECIFICITY:
                continue
            seed_local = local_name(seed.iri)
            for user_iri in self._users_of(seed.iri):
                if user_iri in linked_iris:
                    continue
                for axiom in self._profile(user_iri):
                    # only the axiom naming the seed, not the whole profile
                    if seed_local in axiom_tokens(axiom):
                        records.append(self._record(
                            axiom, seed, "rückverweis", intent, hops=1,
                            source_iri=user_iri, protected=True))
        return records

    @staticmethod
    def _merge(records: list[AxiomRecord]) -> list[AxiomRecord]:
        best = {}
        for record in records:
            current = best.get(record.axiom)
            if current is None:
                best[record.axiom] = record
                continue
            winner = record if record.score > current.score else current
            winner.protected = current.protected or record.protected
            winner.reserved = current.reserved or record.reserved
            best[record.axiom] = winner
        return list(best.values())

    @staticmethod
    def _only_linked(records: list[AxiomRecord],
                     linked_names: set) -> list[AxiomRecord]:
        kept = []
        dropped = 0
        for record in records:
            if any(token in linked_names for token in axiom_tokens(record.axiom)):
                kept.append(record)
            else:
                dropped += 1
        if dropped:
            logger.info("   [Retrieval] %d Axiome ohne verlinkte Entität "
                        "verworfen.", dropped)
        return kept

    def _select(self, records: list[AxiomRecord],
                linked_names: set) -> list[AxiomRecord]:
        ordered = sorted(records,
                         key=lambda record: self._tie_key(record, linked_names))

        reserved = [record for record in ordered if record.reserved]
        protected = [record for record in ordered if record.protected]
        scored = [record for record in ordered
                  if record.score >= self.MIN_AXIOM_SCORE]
        per_kind = []
        for kind in unique(record.kind for record in ordered):
            same_kind = [record for record in ordered if record.kind == kind]
            per_kind.extend(same_kind[:self.MIN_PER_KIND])

        selected = []
        chosen = set()
        for group in (reserved, per_kind, protected, scored):
            for record in group:
                if record.axiom not in chosen:
                    chosen.add(record.axiom)
                    selected.append(record)

        dropped = len(ordered) - len(selected)
        if dropped:
            logger.info("   [Retrieval] %d Axiome unter der Score-Grenze "
                        "verworfen.", dropped)
        selected.sort(key=lambda record: self._tie_key(record, linked_names))
        return selected

    def _tie_key(self, record: AxiomRecord, linked_names: set) -> tuple:
        return (0 if record.source_entity in linked_names else 1,
                record.hops,
                KIND_PRIORITY.get(record.kind, 9),
                -record.score,
                record.axiom)

    @staticmethod
    def _subclass_counts(records) -> dict[str, set]:
        listed = {}
        for record in records:
            if record.rule != "subklassen":
                continue
            match = SUBCLASS_AXIOM.match(record.axiom)
            if match:
                child, parent = match.groups()
                listed.setdefault(parent, set()).add(child)
        return listed

    def _subclass_summaries(self, listed: dict[str, set]) -> list[str]:
        lines = []
        for parent in sorted(listed):
            iri = self.iri_by_local.get(parent)
            if not iri:
                continue
            total = len(self._children(iri))
            # only give a count if the list is complete
            if total and total == len(listed[parent]):
                lines.append("n_subclasses(" + parent + ") = " + str(total))
        return lines

    def _completeness(self, listed: dict[str, set],
                      enumeration_parents: set) -> list[dict]:
        report = []
        for parent in sorted(enumeration_parents):
            iri = self.iri_by_local.get(parent)
            if not iri:
                continue
            total = len(self._children(iri))
            count = len(listed.get(parent, set()))
            report.append({"parent": parent, "total": total, "listed": count,
                           "complete": count >= total})
        return report

    def _enumeration_targets(self, seeds, records) -> list[str]:
        seed_locals = {local_name(link.iri) for link in seeds}
        targets = []
        for record in records:
            if record.rule != "subklassen":
                continue
            parent, child = self._parent_and_child(record.axiom)
            if parent is None or parent not in seed_locals:
                continue
            if child not in targets:
                targets.append(child)
        return targets

    def _build_property_frames(self) -> dict[str, list[str]]:
        frames = {}
        for iri, info in self.facts.items():
            if info.get("type") not in PROPERTIES:
                continue
            for class_iri in info.get("domain", []) + info.get("range", []):
                entry = frames.setdefault(class_iri, [])
                if iri not in entry:
                    entry.append(iri)
        return frames

    def _record(self, axiom: str, link: LinkResult, rule: str, intent: str,
                hops: int, source_iri: str, protected: bool = False,
                reserved: bool = False) -> AxiomRecord:
        kind = axiom_kind(axiom)
        weight = self.RULE_WEIGHTS[rule]
        type_weight = TYPE_WEIGHTS[intent].get(kind, 1.0)
        return AxiomRecord(
            axiom=axiom, score=round(weight * type_weight * link.score, 4),
            rule=rule, hops=hops, source_entity=local_name(source_iri),
            kind=kind, protected=protected, reserved=reserved)

    def _profile(self, iri: str) -> list[str]:
        return KnowledgeGraph.axioms_for_entity(iri, self.facts)

    def _children(self, iri: str) -> list[str]:
        return sorted(self.facts.get(iri, {}).get("subclasses", []))

    def _users_of(self, iri: str) -> list[str]:
        users = list(self.filler_users.get(iri, []))
        for child in self._children(iri):
            users.extend(self.filler_users.get(child, []))
        return sorted(set(users))

    @staticmethod
    def _unique_links(links) -> list[LinkResult]:
        seen = set()
        result = []
        for link in links:
            if link.iri not in seen:
                seen.add(link.iri)
                result.append(link)
        return result

    @staticmethod
    def _parent_and_child(axiom: str):
        match = SUBCLASS_AXIOM.match(axiom)
        if match:
            child, parent = match.groups()
            return parent, child
        if axiom.startswith("ClassAssertion(") and axiom.endswith(")"):
            parts = axiom[len("ClassAssertion("):-1].split(",", 1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
        return None, None

    @staticmethod
    def _empty_result(intent: str) -> dict:
        return {"axioms": [], "intent": intent, "context_summaries": [],
                "seeds": [],
                "completeness": [], "enumeration_targets": []}

    @staticmethod
    def _log_selection(selected: list[AxiomRecord]) -> None:
        counts = {}
        for record in selected:
            counts[record.kind] = counts.get(record.kind, 0) + 1
        detail = ", ".join(kind + "=" + str(count)
                           for kind, count in sorted(counts.items()))
        logger.info("   [Retrieval] %d Axiome im Kontext (%s).", len(selected),
                    detail or "leer")
