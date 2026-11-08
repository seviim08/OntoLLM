import re


KIND_TAXONOMY = "taxonomy"  # SubClassOf(A, B) with named B
KIND_RESTRICTION = "restriction"  # SubClassOf(A, p some X)
KIND_EQUIVALENCE = "equivalence"  # EquivalentClasses(A, ...)
KIND_PROPERTY = "property_frame"  # domain, range, inverse, characteristics
KIND_ASSERTION = "assertion"  # ABox

PROPERTY_KEYWORDS = (
    "ObjectPropertyDomain", "ObjectPropertyRange", "DataPropertyDomain",
    "DataPropertyRange", "InverseObjectProperties", "SubObjectPropertyOf",
    "SubDataPropertyOf",
)

CHARACTERISTIC_KEYWORDS = (
    "FunctionalProperty", "InverseFunctionalProperty", "TransitiveProperty",
    "SymmetricProperty", "AsymmetricProperty", "ReflexiveProperty",
    "IrreflexiveProperty",
)

MANCHESTER_KEYWORDS = {
    "some", "only", "and", "or", "not", "value", "min", "max", "exactly",
    "that", "self",
}
QUANTIFIER_WORDS = {"some", "only", "value", "min", "max", "exactly"}

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")

OPAQUE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9]*_[0-9]+$")  # e.g. DOID_10459


def axiom_kind(axiom: str) -> str:
    head, args = split_axiom(axiom)
    if head in PROPERTY_KEYWORDS or head in CHARACTERISTIC_KEYWORDS:
        return KIND_PROPERTY
    if head == "EquivalentClasses":
        return KIND_EQUIVALENCE
    if head == "SubClassOf":
        filler = args[-1] if args else ""
        for quantifier in QUANTIFIER_WORDS:
            if " " + quantifier + " " in " " + filler + " ":
                return KIND_RESTRICTION
        for operator in (" and ", " or ", "not ", "{"):
            if operator in filler:
                return KIND_RESTRICTION
        return KIND_TAXONOMY
    return KIND_ASSERTION


def axiom_tokens(axiom: str) -> list[str]:
    tokens = []
    for token in TOKEN_PATTERN.findall(axiom):
        if token not in MANCHESTER_KEYWORDS:
            tokens.append(token)
    return tokens


def build_label_index(facts: dict[str, dict]) -> dict[str, str]:
    index = {}
    for iri, info in facts.items():
        labels = []
        for label in info.get("labels", []):
            if label.strip():
                labels.append(label.strip())
        if not labels:
            continue
        # the shortest label is usually the main name
        label = sorted(labels, key=lambda text: (len(text), text))[0]
        local = local_name(iri)
        if label.lower() != humanize_local_name(local):
            index[local] = label

    occurrences = {}
    for label in index.values():
        occurrences[label] = occurrences.get(label, 0) + 1
    return {local: label for local, label in index.items()
            if occurrences[label] == 1}


def annotate_axiom(axiom: str, label_index: dict[str, str]) -> str:
    if not label_index:
        return axiom

    def replace(match: re.Match) -> str:
        token = match.group(0)
        label = label_index.get(token)
        if token in MANCHESTER_KEYWORDS or not label:
            return token
        return label + " [" + token + "]"

    return TOKEN_PATTERN.sub(replace, axiom)


def unresolved_ids(axiom: str, label_index: dict[str, str],
                   known_names) -> list[str]:
    missing = []
    for token in axiom_tokens(axiom):
        if not OPAQUE_ID.match(token):
            continue
        if label_index.get(token) or token in known_names:
            continue
        if token not in missing:
            missing.append(token)
    return missing


def describe_entities(iris, facts: dict[str, dict],
                      label_index: dict[str, str]) -> list[str]:
    lines = []
    for iri in unique(iris):
        info = facts.get(iri)
        if not info or not info.get("descriptions"):
            continue
        local = local_name(iri)
        label = label_index.get(local)
        name = label + " [" + local + "]" if label else local
        lines.append("- " + name + ": " + " ".join(info["descriptions"]))
    return lines


def split_axiom(axiom: str) -> tuple[str, list[str]]:
    text = axiom.strip()
    if "(" not in text or not text.endswith(")"):
        return text, []
    head, body = text.split("(", 1)
    return head, split_arguments(body[:-1])


def split_arguments(body: str) -> list[str]:
    parts = []
    current = ""
    depth = 0
    for char in body:
        if char in "({":
            depth += 1
        elif char in ")}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    parts.append(current.strip())
    return [part for part in parts if part]


def local_name(iri: str) -> str:
    if "#" in iri:
        return iri.split("#")[-1]
    if "/" in iri:
        return iri.rstrip("/").split("/")[-1]
    return iri


def humanize_local_name(name: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)
    spaced = spaced.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", spaced).strip().lower()


def unique(values):
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
