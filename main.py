import json
import logging
import os
from datetime import datetime

from pipeline import OntoQAPipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CONFIG_NAME = "config_doid.json"  # config_pizza.json or config_doid.json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_DIR = os.path.join(DATA_DIR, "config")
ONT_DIR = os.path.join(DATA_DIR, "ontologies")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")


if __name__ == "__main__":
    with open(os.path.join(CONFIG_DIR, CONFIG_NAME), encoding="utf-8") as f:
        config = json.load(f)

    ontology_file_path = os.path.join(ONT_DIR,
                                 os.path.basename(config["ontology_file"]))
    dependencies_paths = []
    for local_path, uri in config.get("dependencies", []):
        dependencies_paths.append([os.path.join(ONT_DIR,
                                          os.path.basename(local_path)), uri])
    reasoner_name = config.get("reasoner", "")

    pipeline = OntoQAPipeline(ontology_file_path, dependencies_paths=dependencies_paths,
                              reasoner_name=reasoner_name)
    ontology_name = os.path.splitext(os.path.basename(ontology_file_path))[0]
    logger.info("Ontology: %s", ontology_name)

    # always stop the JVM, even if a question fails
    pipeline.initialize()
    try:
        results = pipeline.run_questions(config.get("questions", []))
    finally:
        pipeline.cleanup()

    output_dir_path = os.path.join(OUTPUT_DIR, config.get("output_dir", ""))
    os.makedirs(output_dir_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    output_path = os.path.join(
        output_dir_path, "result_" + ontology_name + "_" + run_id + ".json")
    payload = {
        "reasoner": reasoner_name or "none",
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Ergebnisse gespeichert: %s", output_path)
