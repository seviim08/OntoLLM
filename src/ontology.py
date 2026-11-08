import logging
import math
import os
import tempfile
from pathlib import Path

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from .axioms import (CHARACTERISTIC_KEYWORDS, humanize_local_name, local_name,
                     unique)

logger = logging.getLogger(__name__)


CLASS = "Class"
OBJECT_PROPERTY = "ObjectProperty"
DATA_PROPERTY = "DataProperty"
INDIVIDUAL = "Individual"
PROPERTIES = (OBJECT_PROPERTY, DATA_PROPERTY)

RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
OWL_NS = "http://www.w3.org/2002/07/owl#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

OWL_THING = OWL_NS + "Thing"
OWL_NOTHING = OWL_NS + "Nothing"
OWL_NAMED_INDIVIDUAL = OWL_NS + "NamedIndividual"

SKIP_CLS = {OWL_THING, OWL_NOTHING}
SKIP_PROP = {OWL_NS + "topObjectProperty", OWL_NS + "bottomObjectProperty",
             OWL_NS + "topDataProperty", OWL_NS + "bottomDataProperty"}

CHARACTERISTIC_IRIS = {OWL_NS + name for name in CHARACTERISTIC_KEYWORDS}


class OntologyLoader:
    def load(self, main_file: str, dependencies=None):
        if dependencies is None:
            dependencies = []
        main_path = os.path.abspath(main_file)
        if not os.path.exists(main_path):
            raise FileNotFoundError("Ontologie nicht gefunden: " + main_path)

        tmp = tempfile.TemporaryDirectory()
        sources = list(dependencies) + [(main_path, None)]

        uri_mapping = {}
        copies = []
        for local_path, web_uri in sources:
            original = os.path.abspath(local_path)
            if not os.path.exists(original):
                logger.warning("Datei fehlt – %s", original)
                continue
            target = os.path.join(tmp.name, os.path.basename(original))
            copies.append((original, target))
            if web_uri is None:
                continue
            target_uri = Path(target).as_uri()
            uri_mapping[web_uri] = target_uri
            # the same URI appears with and without a trailing slash
            uri_mapping[web_uri.rstrip("/")] = target_uri
            uri_mapping[web_uri.rstrip("/") + "/"] = target_uri

        main_temp_path = ""
        for original, target in copies:
            with open(original, encoding="utf-8") as f:
                content = f.read()
            for web_uri, target_uri in uri_mapping.items():
                content = content.replace(web_uri, target_uri)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            if original == main_path:
                main_temp_path = target

        if not main_temp_path:
            tmp.cleanup()
            raise RuntimeError("Haupt-Ontologie konnte nicht ins "
                               "Temp-Verzeichnis kopiert werden.")
        return main_temp_path, tmp


class ReasoningEngine:
    # ELK is fast but ignores only, not and or, HermiT and Pellet handle them
    REASONER = "ELK"

    OUTPUT_FORMAT = "rdfxml"

    INFERENCE_GENERATORS = (
        "InferredClassAssertionAxiomGenerator",
        "InferredSubClassAxiomGenerator",
        "InferredDisjointClassesAxiomGenerator",
        "InferredEquivalentClassAxiomGenerator",
        "InferredEquivalentDataPropertiesAxiomGenerator",
        "InferredEquivalentObjectPropertyAxiomGenerator",
        "InferredInverseObjectPropertiesAxiomGenerator",
        "InferredSubDataPropertyAxiomGenerator",
        "InferredSubObjectPropertyAxiomGenerator",
        "InferredDataPropertyCharacteristicAxiomGenerator",
        "InferredObjectPropertyCharacteristicAxiomGenerator",
    )

    def __init__(self, ontology_path: str, reasoner: str = ""):
        self.ontology_path = ontology_path
        self.reasoner_name = reasoner or self.REASONER
        self.reasoner = None

    def save_inferred(self, output_path: str) -> None:
        if self.reasoner is None:
            from owlapy.owl_reasoner import SyncReasoner  # starts the JVM
            logger.info("Reasoner: %s", self.reasoner_name)
            self.reasoner = SyncReasoner(ontology=self.ontology_path,
                                         reasoner=self.reasoner_name)
        self.reasoner.infer_axioms_and_save(
            output_path=output_path,
            output_format=self.OUTPUT_FORMAT,
            inference_types=list(self.INFERENCE_GENERATORS),
        )
        logger.info("Gereasonte Ontologie gespeichert: %s", output_path)

    def shutdown(self) -> None:
        from owlapy.static_funcs import stopJVM
        stopJVM()


class KnowledgeGraph:
    # IAO_0000115 is the definition annotation used in OBO ontologies
    ANNOTATION_PROPS = (
        "http://www.w3.org/2000/01/rdf-schema#comment",
        "http://purl.obolibrary.org/obo/IAO_0000115",
        "http://www.w3.org/2004/02/skos/core#definition",
    )

    LABEL_LANGUAGES = {"en"}  # keeps foreign labels out of the prompt

    RESTRICTION_QUANTIFIERS = (
        (OWL.someValuesFrom, "some"), (OWL.allValuesFrom, "only"),
        (OWL.hasValue, "value"),
    )
    RESTRICTION_CARDINALITIES = (
        (OWL.minCardinality, "min"), (OWL.minQualifiedCardinality, "min"),
        (OWL.maxCardinality, "max"), (OWL.maxQualifiedCardinality, "max"),
        (OWL.cardinality, "exactly"), (OWL.qualifiedCardinality, "exactly"),
    )

    def __init__(self, ontology_path: str):
        self.graph = Graph()
        self.graph.parse(ontology_path, format="xml")
        logger.info("KnowledgeGraph geladen: %d Tripel.", len(self.graph))

    def build_ontology_facts(self) -> dict[str, dict]:
        facts = {}
        self._collect_classes(facts)
        self._collect_properties(facts, OWL.ObjectProperty, OBJECT_PROPERTY)
        self._collect_properties(facts, OWL.DatatypeProperty, DATA_PROPERTY)
        self._collect_individuals(facts)
        self._collect_annotations(facts)
        logger.info("Faktenindex: %d Entitäten (Klassen, Properties, "
                    "Individuen).", len(facts))
        return facts

    @staticmethod
    def axioms_for_entity(iri: str, facts: dict[str, dict]) -> list[str]:
        info = facts.get(iri)
        if info is None:
            return []
        subject = local_name(iri)
        entity_type = info["type"]
        axioms = []

        if entity_type == INDIVIDUAL:
            for class_iri in info.get("types", []):
                axioms.append("ClassAssertion(" + local_name(class_iri)
                              + ", " + subject + ")")
            for edge in info.get("outgoing", []):
                axioms.append(local_name(edge["prop"]) + "(" + subject + ", "
                              + local_name(edge["target"]) + ")")
            for value in info.get("data_values", []):
                axioms.append("DataPropertyAssertion("
                              + local_name(value["prop"]) + ", " + subject
                              + ", " + value["value"] + ")")

        elif entity_type in PROPERTIES:
            is_object = entity_type == OBJECT_PROPERTY
            sub_keyword = "SubObjectPropertyOf" if is_object else "SubDataPropertyOf"
            domain_keyword = "ObjectPropertyDomain" if is_object else "DataPropertyDomain"
            range_keyword = "ObjectPropertyRange" if is_object else "DataPropertyRange"
            for superprop in info.get("superproperties", []):
                axioms.append(sub_keyword + "(" + subject + ", "
                              + local_name(superprop) + ")")
            for domain in info.get("domain", []):
                axioms.append(domain_keyword + "(" + subject + ", "
                              + local_name(domain) + ")")
            for range_ in info.get("range", []):
                axioms.append(range_keyword + "(" + subject + ", "
                              + local_name(range_) + ")")
            if is_object:
                for inverse in info.get("inverse_of", []):
                    axioms.append("InverseObjectProperties(" + subject + ", "
                                  + local_name(inverse) + ")")
            for characteristic in info.get("characteristics", []):
                axioms.append(characteristic + "(" + subject + ")")

        else:
            for superclass in info.get("superclasses", []):
                axioms.append("SubClassOf(" + subject + ", "
                              + local_name(superclass) + ")")
            for restriction in info.get("restrictions", []):
                axioms.append("SubClassOf(" + subject + ", " + restriction + ")")
            for equivalent in info.get("equivalent_classes", []):
                axioms.append("EquivalentClasses(" + subject + ", "
                              + equivalent + ")")

        return unique(axioms)

    @staticmethod
    def specificity_index(facts: dict[str, dict]) -> dict[str, float]:
        index = {}
        for iri, info in facts.items():
            size = len(KnowledgeGraph.descendants(iri, facts)) \
                + len(info.get("instances", []))
            index[iri] = 1.0 / math.log2(2 + size)
        return index

    def restriction_filler_index(self) -> dict[str, list[str]]:
        index = {}
        for cls in self.graph.subjects(RDF.type, OWL.Class):
            if not isinstance(cls, URIRef) or str(cls) in SKIP_CLS:
                continue
            for predicate in (RDFS.subClassOf, OWL.equivalentClass):
                for expression in self.graph.objects(cls, predicate):
                    if not isinstance(expression, BNode):
                        continue
                    for filler_iri in self._expression_iris(expression):
                        entry = index.setdefault(filler_iri, [])
                        if str(cls) not in entry:
                            entry.append(str(cls))
        return index

    def _collect_classes(self, facts: dict[str, dict]) -> None:
        for cls in self.graph.subjects(RDF.type, OWL.Class):
            if not isinstance(cls, URIRef) or str(cls) in SKIP_CLS:
                continue
            iri = str(cls)
            info = facts.setdefault(iri, {
                "type": CLASS,
                "superclasses": [], "subclasses": [], "equivalent_classes": [],
                "restrictions": [], "instances": [],
            })
            for superclass in self.graph.objects(cls, RDFS.subClassOf):
                if isinstance(superclass, URIRef):
                    if str(superclass) not in SKIP_CLS and superclass != cls:
                        info["superclasses"].append(str(superclass))
                elif isinstance(superclass, BNode):
                    rendered = self.render_class_expression(superclass)
                    if rendered:
                        info["restrictions"].append(rendered)
            for equivalent in self.graph.objects(cls, OWL.equivalentClass):
                if isinstance(equivalent, URIRef):
                    if str(equivalent) not in SKIP_CLS and equivalent != cls:
                        info["equivalent_classes"].append(
                            local_name(str(equivalent)))
                elif isinstance(equivalent, BNode):
                    rendered = self.render_class_expression(equivalent)
                    if rendered:
                        info["equivalent_classes"].append(rendered)

        for iri, info in facts.items():
            cls = URIRef(iri)
            for subclass in self.graph.subjects(RDFS.subClassOf, cls):
                if isinstance(subclass, URIRef) and str(subclass) in facts \
                        and subclass != cls:
                    info["subclasses"].append(str(subclass))
            for individual in self.graph.subjects(RDF.type, cls):
                if isinstance(individual, URIRef) \
                        and str(individual) not in facts \
                        and str(individual) != OWL_NAMED_INDIVIDUAL:
                    info["instances"].append({"iri": str(individual)})

    def _collect_properties(self, facts: dict[str, dict], owl_type: URIRef,
                            type_label: str) -> None:
        is_object = type_label == OBJECT_PROPERTY
        for prop in self.graph.subjects(RDF.type, owl_type):
            if not isinstance(prop, URIRef) or str(prop) in SKIP_PROP:
                continue
            iri = str(prop)
            fields = {
                "type": type_label,
                "superproperties": [], "subproperties": [],
                "domain": [], "range": [], "characteristics": [],
            }
            if is_object:
                fields["inverse_of"] = []
            info = facts.setdefault(iri, fields)
            for superprop in self.graph.objects(prop, RDFS.subPropertyOf):
                if isinstance(superprop, URIRef) \
                        and str(superprop) not in SKIP_PROP and superprop != prop:
                    info["superproperties"].append(str(superprop))
            for subprop in self.graph.subjects(RDFS.subPropertyOf, prop):
                if isinstance(subprop, URIRef) \
                        and str(subprop) not in SKIP_PROP and subprop != prop:
                    info["subproperties"].append(str(subprop))
            for domain in self.graph.objects(prop, RDFS.domain):
                if isinstance(domain, URIRef) and str(domain) != OWL_THING:
                    info["domain"].append(str(domain))
            for range_ in self.graph.objects(prop, RDFS.range):
                if isinstance(range_, URIRef) and str(range_) != OWL_THING:
                    info["range"].append(str(range_))
            if is_object:
                for inverse in self.graph.objects(prop, OWL.inverseOf):
                    if isinstance(inverse, URIRef):
                        info["inverse_of"].append(str(inverse))
            for rdf_type in self.graph.objects(prop, RDF.type):
                if isinstance(rdf_type, URIRef) \
                        and str(rdf_type) in CHARACTERISTIC_IRIS:
                    info["characteristics"].append(local_name(str(rdf_type)))

    def _collect_individuals(self, facts: dict[str, dict]) -> None:
        for individual in self.graph.subjects(RDF.type, OWL.NamedIndividual):
            if not isinstance(individual, URIRef):
                continue
            iri = str(individual)
            info = facts.setdefault(iri, {
                "type": INDIVIDUAL,
                "types": [], "outgoing": [], "data_values": [],
            })
            for rdf_type in self.graph.objects(individual, RDF.type):
                if isinstance(rdf_type, URIRef) and str(rdf_type) not in (
                        OWL_NAMED_INDIVIDUAL, OWL_THING, OWL_NOTHING):
                    info["types"].append(str(rdf_type))
            for prop, target in self.graph.predicate_objects(individual):
                name = str(prop)
                if name == RDF_TYPE or name.startswith(RDFS_NS) \
                        or name.startswith(OWL_NS):
                    continue
                if isinstance(target, URIRef):
                    info["outgoing"].append({"prop": name,
                                             "target": str(target)})
                elif isinstance(target, Literal):
                    info["data_values"].append({"prop": name,
                                                "value": str(target)})

    def _collect_annotations(self, facts: dict[str, dict]) -> None:
        for iri, info in facts.items():
            entity = URIRef(iri)
            labels = []
            for label in self.graph.objects(entity, RDFS.label):
                if self._language_allowed(label):
                    labels.append(str(label))
            info["labels"] = labels

            descriptions = []
            for prop in self.ANNOTATION_PROPS:
                for value in self.graph.objects(entity, URIRef(prop)):
                    if not self._language_allowed(value):
                        continue
                    text = " ".join(str(value).split())
                    if text and text not in descriptions:
                        descriptions.append(text)
            info["descriptions"] = descriptions

            names = set()
            for label in labels:
                if label.strip():
                    names.add(label.strip().lower())
            names.add(humanize_local_name(local_name(iri)))
            info["link_names"] = sorted(names)

    def _language_allowed(self, value) -> bool:
        language = getattr(value, "language", None)
        if not language:
            return True
        return language.lower().split("-")[0] in self.LABEL_LANGUAGES

    def render_class_expression(self, node) -> str:
        if isinstance(node, URIRef):
            return local_name(str(node))
        if isinstance(node, Literal):
            return str(node)
        if not isinstance(node, BNode):
            return ""

        prop = self.graph.value(node, OWL.onProperty)
        if prop is not None:
            return self._render_restriction(node, local_name(str(prop)))

        union = self.graph.value(node, OWL.unionOf)
        if union is not None:
            return "(" + " or ".join(self._render_members(union)) + ")"

        # no outer brackets, matches the expected evidence in the configs
        intersection = self.graph.value(node, OWL.intersectionOf)
        if intersection is not None:
            return " and ".join(self._render_members(intersection))

        complement = self.graph.value(node, OWL.complementOf)
        if complement is not None:
            inner = self.render_class_expression(complement)
            if not inner:
                return ""
            if " " in inner and not inner.startswith("("):
                inner = "(" + inner + ")"
            return "not " + inner

        one_of = self.graph.value(node, OWL.oneOf)
        if one_of is not None:
            return "{" + ", ".join(self._render_members(one_of)) + "}"

        logger.debug("Unbekannter Klassenausdruck: %s", node)
        return ""

    def _render_restriction(self, node: BNode, prop: str) -> str:
        for predicate, keyword in self.RESTRICTION_QUANTIFIERS:
            filler = self.graph.value(node, predicate)
            if filler is not None:
                return prop + " " + keyword + " " \
                    + self.render_class_expression(filler)

        for predicate, keyword in self.RESTRICTION_CARDINALITIES:
            number = self.graph.value(node, predicate)
            if number is None:
                continue
            filler = self.graph.value(node, OWL.onClass)
            if filler is None:
                filler = self.graph.value(node, OWL.onDataRange)
            if filler is None:
                return prop + " " + keyword + " " + str(number)
            return prop + " " + keyword + " " + str(number) + " " \
                + self.render_class_expression(filler)

        logger.debug("Restriktion ohne bekannten Quantor: %s", node)
        return prop

    def _render_members(self, collection) -> list[str]:
        parts = []
        for member in self.graph.items(collection):
            rendered = self.render_class_expression(member)
            if not rendered:
                continue
            if rendered.startswith("not "):
                rendered = "(" + rendered + ")"
            parts.append(rendered)
        return parts

    def _expression_iris(self, node, in_restriction: bool = False) -> set:
        iris = set()
        if isinstance(node, URIRef):
            if in_restriction and str(node) not in SKIP_CLS:
                iris.add(str(node))
            return iris
        if not isinstance(node, BNode):
            return iris

        # only entities behind a quantifier count as fillers
        if self.graph.value(node, OWL.onProperty) is not None:
            for predicate in (OWL.someValuesFrom, OWL.allValuesFrom,
                              OWL.hasValue, OWL.onClass, OWL.onDataRange):
                for filler in self.graph.objects(node, predicate):
                    iris |= self._expression_iris(filler, True)
            return iris

        for predicate in (OWL.unionOf, OWL.intersectionOf, OWL.oneOf):
            for collection in self.graph.objects(node, predicate):
                for member in self.graph.items(collection):
                    iris |= self._expression_iris(member, in_restriction)
        for complement in self.graph.objects(node, OWL.complementOf):
            iris |= self._expression_iris(complement, in_restriction)
        return iris

    @staticmethod
    def descendants(iri: str, facts: dict[str, dict]) -> set:
        found = set()
        queue = [iri]
        while queue:
            current = queue.pop()
            info = facts.get(current, {})
            children = info.get("subclasses") or info.get("subproperties") or []
            for child in children:
                if child not in found:
                    found.add(child)
                    queue.append(child)
        return found
