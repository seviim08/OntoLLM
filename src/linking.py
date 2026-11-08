import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .axioms import axiom_tokens, humanize_local_name, local_name, unique
from .ontology import CLASS, PROPERTIES, KnowledgeGraph
from .stopwords import STOPWORDS

logger = logging.getLogger(__name__)


LINKING_METHOD = "hybrid"  # lexical, dense or hybrid

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

INTENT_DEFINITION = "definition"  # "What is X?", also the fallback
INTENT_ENUMERATION = "enumeration"  # "List all X"
INTENT_VERIFICATION = "verification"  # "Is X a Y?"
INTENT_RELATION = "relation"  # "How are X and Y related?"
INTENT_EXPLANATION = "explanation"  # "Why ...?"

# first match wins
INTENT_PATTERNS = (
    (INTENT_ENUMERATION, re.compile(
        r"^(list|name|enumerate|which|what\s+(kinds|types|sorts))\b|"
        r"\ball\s+(the\s+)?\w+\s+(that|which)\b", re.I)),
    (INTENT_VERIFICATION, re.compile(
        r"^(is|are|does|do|can|could|has|have|was|were)\b", re.I)),
    (INTENT_EXPLANATION, re.compile(r"^(why|how\s+come)\b|\bexplain\b", re.I)),
    (INTENT_RELATION, re.compile(
        r"\b(relation|related|connect|between|part\s+of)\b|"
        r"^what\s+(is|are)\s+the\s+\w+\s+of\b", re.I)),
)

COVERAGE_PATTERN = re.compile(
    r"\b(ontology|knowledge\s*(base|graph)|dataset)\b.{0,40}"
    r"\b(contain|include|cover|know|have|hold)\b|"
    r"\b(contain|include|cover)\b.{0,40}\binformation\b", re.I | re.S)

# only excluded at the end of a mention, so "usher syndrome type 1" stays
META_WORDS = {
    "ontology", "ontologies", "information", "info", "data", "database",
    "knowledge", "class", "classes", "entity", "entities",
    "type", "types", "subtype", "subtypes", "kind", "kinds", "form", "forms",
}

MIN_SINGLE_TOKEN_LEN = 3


@dataclass(frozen=True)
class Mention:
    text: str
    start: int
    end: int

    def length(self) -> int:
        return self.end - self.start

    def contains(self, other: "Mention") -> bool:
        return (self.start <= other.start and other.end <= self.end
                and self.length() > other.length())


@dataclass
class LinkResult:
    mention: str
    iri: str
    score: float
    method: str
    start: int = 0
    end: int = 0
    ic: float = 0.0


def detect_intent(question: str) -> str:
    sentences = sentences_of(question)
    for intent, pattern in INTENT_PATTERNS:
        for sentence in sentences:
            if pattern.search(sentence):
                return intent
    return INTENT_DEFINITION


def is_coverage_question(question: str) -> bool:
    return bool(COVERAGE_PATTERN.search(question))


def sentences_of(question: str) -> list[str]:
    parts = []
    for part in re.split(r"[.!?]+", question):
        if part.strip():
            parts.append(part.strip())
    return parts or [question.strip()]


def extract_mentions(question: str, ngram_max: int, known_names: set,
                     interior: set) -> list[Mention]:
    mentions = []
    for segment in noun_phrase_segments(question, interior):
        for size in range(1, ngram_max + 1):
            for offset in range(0, len(segment) - size + 1):
                part = segment[offset:offset + size]
                tokens = [token for _, token in part]
                phrase = " ".join(tokens)
                if phrase not in known_names and not is_plausible_mention(tokens):
                    continue
                mentions.append(Mention(phrase, part[0][0], part[-1][0] + 1))
    return mentions


def noun_phrase_segments(question: str, interior: set) -> list[list[tuple]]:
    segments = []
    current = []
    index = 0
    for match in re.finditer(r"\w+|[^\w\s]", question.lower()):
        token = match.group(0)
        if not token[0].isalnum():
            segments.append(current)
            current = []
            index += 1
            continue
        if token in STOPWORDS and token not in interior:
            segments.append(current)
            current = []
        else:
            current.append((index, token))
        index += 1
    segments.append(current)
    return [segment for segment in segments if segment]


def is_plausible_mention(tokens: list[str]) -> bool:
    if any(token in STOPWORDS for token in tokens):
        return False
    if tokens[-1] in META_WORDS:
        return False
    if all(token in META_WORDS for token in tokens):
        return False
    if len(tokens) == 1 and len(tokens[0]) < MIN_SINGLE_TOKEN_LEN:
        return False
    return True


def interior_stopwords(known_names) -> set:
    inside = set()
    for name in known_names:
        tokens = re.findall(r"\w+", name.lower())
        for token in tokens[1:-1]:
            if token in STOPWORDS:
                inside.add(token)
    return inside


def is_subphrase(short: str, long: str) -> bool:
    return short != long and (" " + short + " ") in (" " + long + " ")


def known_link_names(facts: dict[str, dict]) -> set:
    names = set()
    for iri, info in facts.items():
        if is_domain_entity(iri):
            names.update(info.get("link_names", []))
    return names


def is_domain_entity(iri: str) -> bool:
    return not local_name(iri).startswith("_")


class EntityLinker:
    METHODS = ("lexical", "dense", "hybrid")

    LEXICAL_THRESHOLD = 0.80  # still matches "pizzas", but not "vegetarier"

    DENSE_MODEL = "intfloat/multilingual-e5-large"
    DENSE_THRESHOLD = 0.85
    # prefixes required by e5
    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

    IC_KEEP_SHARE = 0.9  # share of the highest IC a link needs to stay

    CHECK_LIMIT = 5

    def __init__(self, facts: dict[str, dict]):
        if LINKING_METHOD not in self.METHODS:
            raise ValueError("Unbekannte Entity-Linking-Methode: "
                             + repr(LINKING_METHOD)
                             + " (erlaubt: " + ", ".join(self.METHODS) + ")")
        self.method = LINKING_METHOD
        self.facts = facts
        self._build_name_index()
        self._build_full_text_index()
        self._build_frequency_index()
        self.interior = interior_stopwords(self.known_names)
        # the longest name limits the mention length
        lengths = [len(re.findall(r"\w+", name)) for name in self.names]
        self.ngram_max = max(2, min(6, max(lengths, default=2)))
        if self.method != "lexical":
            self.encoder = self._default_encoder()
            self.name_matrix = self._encode_names()

    def link(self, question: str) -> dict:
        mentions = extract_mentions(question, self.ngram_max,
                                    self.known_names, self.interior)
        texts = unique([mention.text for mention in mentions])
        hits = {}
        if self.method in ("lexical", "hybrid"):
            hits.update(self._lexical_hits(texts))
        if self.method in ("dense", "hybrid"):
            for text, dense in self._dense_hits(texts).items():
                if text not in hits or dense[2] > hits[text][2]:
                    hits[text] = dense

        candidates = []
        for mention in mentions:
            hit = hits.get(mention.text)
            if hit is not None:
                iri, _name, score, method = hit
                candidates.append(LinkResult(mention.text, iri, score, method,
                                             mention.start, mention.end))

        for candidate in candidates:
            candidate.ic = round(self.information_content(candidate.iri), 3)

        links = self._best_per_entity(self._drop_common(
            self._drop_covered(self._best_per_mention(candidates))))
        for link in links:
            logger.info("   [%s] '%s' → %s  (score=%.3f, %s, IC=%.2f)",
                        self.method, link.mention, local_name(link.iri),
                        link.score, link.method, link.ic)
        if not links:
            logger.info("   [%s] keine Entität über Schwellwert verlinkt.",
                        self.method)

        report = {
            "links": links,
            "unlinked_terms": self._open_mentions(question, links),
            "existence_checks": [],
            "intent": detect_intent(question),
            "coverage_question": is_coverage_question(question),
            "seeds": [],
        }
        logger.info("   [Linking] Intention: %s%s", report["intent"],
                    " (Abdeckungsfrage)" if report["coverage_question"] else "")
        self._check_existence(question, report)
        self._collect_seeds(report)
        return report

    def information_content(self, iri: str) -> float:
        # IC(w) = -log2(p(w)) with Laplace smoothing
        count = self.frequency.get(local_name(iri), 0)
        probability = (count + 1) / (self.frequency_total
                                     + len(self.frequency) + 1)
        return -math.log2(probability)

    def lexical_exists(self, term: str) -> dict:
        needle = term.strip().lower()
        if not needle:
            return {"term": term, "found": False, "candidates": []}

        hits = {}
        for text, iri, source in self.full_text:
            if iri not in hits and needle in text:
                hits[iri] = source

        if not hits:
            logger.info("   [Existenzprüfung] '%s': kein Treffer → belegte "
                        "Nichtexistenz.", term)
            return {"term": term, "found": False, "candidates": []}

        ordered = sorted(hits.items(),
                         key=lambda pair: (pair[1] != "link_name",
                                           len(local_name(pair[0]))))
        ordered = ordered[:self.CHECK_LIMIT]
        logger.info("   [Existenzprüfung] '%s': %d Kandidat(en) im Volltext.",
                    term, len(hits))
        return {"term": term, "found": True,
                "candidates": [local_name(iri) for iri, _ in ordered]}

    def _build_name_index(self) -> None:
        self.names = []
        self.iris = []
        self.by_name = {}
        for iri, info in self.facts.items():
            if not is_domain_entity(iri):
                continue
            for name in info.get("link_names", []):
                self.names.append(name)
                self.iris.append(iri)
                if name not in self.by_name:
                    self.by_name[name] = iri
        self.known_names = set(self.by_name)

    def _build_full_text_index(self) -> None:
        self.full_text = []
        for iri, info in self.facts.items():
            self.full_text.append((local_name(iri).lower(), iri, "iri"))
            self.full_text.append((humanize_local_name(local_name(iri)),
                                   iri, "iri"))
            for label in info.get("labels", []):
                self.full_text.append((label.lower(), iri, "label"))
            for name in info.get("link_names", []):
                self.full_text.append((name, iri, "link_name"))
            for description in info.get("descriptions", []):
                self.full_text.append((description.lower(), iri, "description"))

    def _build_frequency_index(self) -> None:
        self.frequency = {}
        for iri in self.facts:
            for axiom in KnowledgeGraph.axioms_for_entity(iri, self.facts):
                for token in set(axiom_tokens(axiom)):
                    self.frequency[token] = self.frequency.get(token, 0) + 1
        self.frequency_total = sum(self.frequency.values())

    def _lexical_hits(self, texts: list[str]) -> dict:
        hits = {}
        for text in texts:
            if text in self.by_name:
                hits[text] = (self.by_name[text], text, 1.0, "exact")
                continue
            match = process.extractOne(text, self.names,
                                       scorer=Levenshtein.normalized_similarity,
                                       score_cutoff=self.LEXICAL_THRESHOLD)
            if match is not None:
                name, score, position = match
                hits[text] = (self.iris[position], name, float(score), "lexical")
        return hits

    def _dense_hits(self, texts: list[str]) -> dict:
        if not texts or not self.names:
            return {}
        queries = [self.QUERY_PREFIX + text for text in texts]
        query_matrix = self._normalize(self.encoder(queries))
        # rows are normalized, so this is the cosine similarity
        similarities = query_matrix.dot(self.name_matrix.T)

        hits = {}
        for row, text in enumerate(texts):
            position = int(np.argmax(similarities[row]))
            score = float(similarities[row][position])
            if score >= self.DENSE_THRESHOLD:
                hits[text] = (self.iris[position], self.names[position],
                              score, "dense")
        return hits

    def _default_encoder(self):
        # imported here because lexical linking does not need it
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.DENSE_MODEL)

        def encode(texts):
            return np.asarray(model.encode(list(texts), show_progress_bar=False))

        return encode

    def _encode_names(self):
        passages = [self.PASSAGE_PREFIX + name for name in self.names]
        key = self.DENSE_MODEL + "\n" + "\n".join(passages)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        cache_path = os.path.join(CACHE_DIR, "link_embeddings_" + digest + ".npz")
        if os.path.exists(cache_path):
            logger.info("   Dense-Embeddings aus Cache: %s", cache_path)
            return np.load(cache_path)["embeddings"]
        logger.info("   Kodiere %d Ontologie-Bezeichnungen vorab (%s) ...",
                    len(passages), self.DENSE_MODEL)
        matrix = self._normalize(self.encoder(passages))
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=matrix)
        return matrix

    @staticmethod
    def _normalize(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    @staticmethod
    def _best_per_mention(candidates: list[LinkResult]) -> list[LinkResult]:
        best = {}
        for candidate in candidates:
            key = candidate.mention
            current = best.get(key)
            if current is None or candidate.score > current.score or (
                    candidate.score == current.score
                    and candidate.iri < current.iri):
                best[key] = candidate
        return list(best.values())

    @staticmethod
    def _drop_covered(links: list[LinkResult]) -> list[LinkResult]:
        kept = []
        for link in links:
            better = None
            for other in links:
                if not is_subphrase(link.mention, other.mention):
                    continue
                same_place = other.start <= link.start and link.end <= other.end
                if same_place:
                    is_better = other.score >= link.score
                else:
                    is_better = other.score > link.score
                if is_better:
                    better = other
                    break
            if better is None:
                kept.append(link)
            else:
                logger.info("   [Linking] '%s' verworfen – steckt in '%s' "
                            "(score %.3f gegen %.3f).", link.mention,
                            better.mention, better.score, link.score)
        return kept

    def _drop_common(self, links: list[LinkResult]) -> list[LinkResult]:
        if not links:
            return links
        best = max(link.ic for link in links)
        threshold = self.IC_KEEP_SHARE * best
        kept = []
        for link in links:
            if link.ic >= threshold:
                kept.append(link)
            else:
                logger.info("   [Linking] '%s' verworfen – zu häufig "
                            "(IC %.2f unter Schwelle %.2f).",
                            link.mention, link.ic, threshold)
        return kept

    @staticmethod
    def _best_per_entity(links: list[LinkResult]) -> list[LinkResult]:
        best = {}
        for link in links:
            current = best.get(link.iri)
            if current is None or link.score > current.score:
                best[link.iri] = link
        return sorted(best.values(), key=lambda link: -link.score)

    def _check_existence(self, question: str, report: dict) -> None:
        to_check = unique(report["unlinked_terms"]
                          + self._open_mentions(question, report["links"],
                                                classes_only=True))
        for term in to_check:
            report["existence_checks"].append(self.lexical_exists(term))

    def _open_mentions(self, question: str, links: list[LinkResult],
                       classes_only: bool = False) -> list[str]:
        linked = []
        linked_texts = set()
        for link in links:
            if not classes_only or self.facts.get(link.iri, {}).get("type") == CLASS:
                linked.append(Mention(link.mention, link.start, link.end))
                linked_texts.add(link.mention)

        free = []
        for mention in extract_mentions(question, self.ngram_max,
                                        self.known_names, self.interior):
            if mention.text in linked_texts:
                continue
            if any(mention.start < other.end and other.start < mention.end
                   for other in linked):
                continue
            free.append(mention)

        # only the longest mentions, not "pizza" in "pizza topping"
        terms = []
        for mention in free:
            if not any(other.contains(mention) for other in free):
                terms.append(mention.text)
        return unique(terms)[:self.CHECK_LIMIT]

    def _collect_seeds(self, report: dict) -> None:
        properties = []
        for link in report["links"]:
            if self.facts.get(link.iri, {}).get("type") in PROPERTIES:
                properties.append(local_name(link.iri))
            else:
                report["seeds"].append(link)
        logger.info("   [Seeds] %s | Properties: %s",
                    [local_name(l.iri) for l in report["seeds"]], properties)
