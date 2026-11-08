# OntoLLM

Dieses Projekt ist im Rahmen meiner Masterthesis entstanden. Ziel war es, mit einer Ontologie als Wissensbasis und Retrieval-Augmented Generation (RAG) ein Frage-Antwort-System zu bauen, das nicht halluziniert und sich bei jeder Antwort nachweisbar auf die Wissensbasis stützt.

## Funktionsweise

Eine Frage durchläuft vier Stufen:

1. **Wissensbasis aufbauen:** Die Ontologie wird geladen und von einem Reasoner materialisiert, sodass auch abgeleitetes Wissen als Axiom vorliegt. Daraus entsteht ein Faktenindex über alle Klassen, Properties und Individuen. Das passiert einmal pro Ontologie.
2. **Entity Linking:** Die Begriffe der Frage werden lexikalisch (Levenshtein) und über Embeddings (Kosinus) den Entitäten der Ontologie zugeordnet. Außerdem wird die Art der Frage erkannt (Definition, Aufzählung, Ja/Nein, Beziehung, Erklärung).
3. **Retrieval:** Aus den zugeordneten Entitäten werden die passenden Axiome gesammelt, bewertet und als Kontext ausgewählt. Jedes Axiom im Kontext muss eine der zugeordneten Entitäten enthalten.
4. **Generierung:** Das Sprachmodell bekommt die Axiome nummeriert und muss die Nummern, auf die sich seine Antwort stützt, als Evidenz angeben. Antworten ohne Evidenz oder mit Entitäten, die nicht aus der Ontologie stammen, werden abgelehnt. Gibt die Wissensbasis keine Antwort her, verweigert das System die Antwort, statt zu raten.

Getestet wurde mit der Pizza-Ontologie und der Disease Ontology (DOID), mit je 14 Fragen samt erwarteter Antwort und Evidenz. `evaluate.py` berechnet daraus pro Frage einen F1-Wert für die Antwort und einen für die Evidenz.

## Starten

Benötigt werden Python, Java (für den Reasoner über owlapy) und [Ollama](https://ollama.com) mit dem Modell `gemma2:27b`.

```
pip install rdflib owlapy rapidfuzz numpy sentence-transformers openai
ollama pull gemma2:27b
python main.py
python evaluate.py
```

Die Ontologien sind wegen ihrer Größe nicht im Repo und gehören nach `data/ontologies/`: [pizza.owl](https://protege.stanford.edu/ontologies/pizza/pizza.owl) und [doid.owl](http://purl.obolibrary.org/obo/doid.owl). Welche Ontologie verwendet wird, steht in `main.py` unter `CONFIG_NAME`.

## Aufbau

```
main.py                     Startet einen Lauf mit der gewählten Konfiguration
pipeline.py                 Verbindet die vier Stufen
evaluate.py                 F1-Auswertung von Antwort und Evidenz
src/
  ontology.py               Laden, Reasoning und Faktenindex
  linking.py                Entity Linking und Erkennung der Fragenart
  retrieval.py              Auswahl der Axiome für den Kontext
  generation.py             Prompt, Aufruf des Sprachmodells und Prüfung der Antwort
  axioms.py                 Hilfsfunktionen für Axiome
  stopwords.py              Stoppwortliste
data/config/                Konfigurationen mit Ontologie, Reasoner und Testfragen
```
