# Sprechertext (Final) - Fahrerzustand-Klassifikation

Geplante Gesamtdauer: ca. 10-12 Minuten

---

## Folie 1 - Titel (ca. 20-30 Sek.)
Hallo zusammen, ich bin Burak Güldüz und ich präsentiere meine Arbeit zur Fahrerzustand-Klassifikation.  
Ziel war es, Fahrerzustände aus multimodalen Sensordaten automatisch in drei Klassen einzuordnen: Attentive, Inattentive und Aggressive.

---

## Folie 2 - Motivation & Problemstellung (ca. 50 Sek.)
Die Motivation ist klar: menschliches Fehlverhalten ist einer der Hauptfaktoren bei Verkehrsunfällen.  
Viele bestehende Systeme nutzen aber nur einfache Signale, zum Beispiel Lenkradkontakt.  
Unser Problem war zusätzlich schwieriger, weil wir im Datensatz keine manuellen Ground-Truth-Labels hatten.  
Das heißt: Wir mussten erst eine robuste Labeling-Strategie entwickeln, bevor wir überhaupt supervised lernen konnten.

---

## Folie 3 - Ziele dieser Arbeit (ca. 45 Sek.)
Die Arbeit hatte vier technische Kernziele:  
erstens ein nachvollziehbares Absolute-Threshold-Labeling, zweitens temporale Glättung gegen Label-Flackern, drittens multimodale Feature-Fusion mit 37 Features und viertens eine End-to-End-Pipeline von Rohdaten bis zur Vorhersage.  
Zusätzlich habe ich supervised und unsupervised Ansätze verglichen.

---

## Folie 4 - Datenerfassung & Modalitäten (ca. 50 Sek.)
Wir nutzen Daten aus dem VTI-Simulator.  
Die Modalitäten sind Fahrzeugdynamik, Eye-Tracking und Physiologie, ergänzt um Umfelddaten.  
Konkret sind das zum Beispiel Beschleunigung, Jerk, Lenkwinkel, Blickrichtung, Augenlidentöffnung sowie Herzrate und RR-Intervalle.  
Durch diese Kombination können wir nicht nur Fahrverhalten, sondern auch Aufmerksamkeitsverhalten abbilden.

---

## Folie 5 - Feature Engineering (ca. 50 Sek.)
Aus den synchronisierten Rohsignalen wurden 37 Features berechnet.  
Für aggressive Fahrweise sind vor allem Jerk, Lenkgeschwindigkeit und Headway-relevante Größen wichtig.  
Für Inattention sind Blickabwendung, Spurabweichung, NDRT-Fehler und Augenlidentöffnung zentral.  
Die Features wurden überwiegend über Rolling Windows berechnet, damit kurzfristiges Rauschen reduziert wird.

---

## Folie 6 - Label-Generierung (ca. 70 Sek.)
Da keine Ground-Truth vorlag, nutze ich ein mehrstufiges Labeling:  
zuerst datengetriebene globale Schwellenwerte, dann kritische Events, danach ein akkumulatives Scoring aus mehreren schwächeren Indikatoren.  
Anschließend kommt Temporal Persistence, hier mit einem 2-Sekunden-Fenster, damit kurze Peaks nicht sofort harte Labelwechsel erzeugen.  
So entstehen stabilere Pseudo-Labels mit klarer Logik.

---

## Folie 7 - Label-Verteilung (ca. 30-40 Sek.)
Hier sieht man die resultierende Klassenverteilung.  
Die Klassen sind nicht perfekt gleich verteilt, aber ausreichend vertreten.  
Für das Training wurde deshalb mit `class_weight='balanced'` gearbeitet, um Verzerrungen durch Imbalance zu reduzieren.

---

## Folie 8 - Modellierung: Random Forest (ca. 45 Sek.)
Random Forest habe ich als Baseline eingesetzt.  
Die wichtigsten Einstellungen sind hier 200 Bäume, begrenzte Tiefe und balancierte Klassengewichte.  
Der Vorteil ist Robustheit und gute Interpretierbarkeit, aber bei komplexen nichtlinearen Zusammenhängen ist RF oft schwächer als Boosting.

---

## Folie 9 - Random Forest Ergebnisse (ca. 50 Sek.)
Die RF-Ergebnisse im gezeigten Split sind:  
Train Accuracy 88.87%, Test Accuracy 88.42%, Cohen’s Kappa 0.8129.  
Die Per-Class-F1-Werte liegen zwischen 0.89 und 0.92.  
Damit ist die Baseline solide, aber für das Projektziel noch nicht optimal.

---

## Folie 10 - RF Ergebnis-Einordnung (ca. 35-45 Sek.)
Bei der Fehleranalyse sieht man, dass RF häufiger zwischen Inattentive und Aggressive verwechselt.  
Gerade in Grenzbereichen reichen einzelne Baumabstimmungen nicht immer aus, um subtile Muster sauber zu trennen.  
Deshalb war der nächste Schritt ein stärkeres Ensemble mit sequentieller Fehlerkorrektur.

---

## Folie 11 - Modellierung: LightGBM (ca. 50 Sek.)
LightGBM nutzt Gradient Boosting und korrigiert Fehler schrittweise über die Baumfolge.  
Wichtige Parameter hier sind unter anderem 200 Estimators, max_depth 6 und learning_rate 0.05.  
Zusätzlich ist GPU-Nutzung möglich, was bei großen Datenmengen die Trainingszeit verbessert.

---

## Folie 12 - LightGBM Ergebnisse & Vergleich (ca. 60 Sek.)
LightGBM erreicht im gezeigten Lauf:  
Train Accuracy 92.48%, Test Accuracy 91.96%, Cohen’s Kappa 0.8697.  
Die F1-Werte liegen bei 0.94 für Attentive und je 0.90 für Inattentive und Aggressive.  
Damit ist LightGBM klar besser als Random Forest, sowohl in Genauigkeit als auch in Übereinstimmung über Zufall hinaus.

---

## Folie 13 - Confusion Matrix (ca. 50 Sek.)
Die Confusion Matrix zeigt die Fehlerstruktur.  
Attentive wird sehr zuverlässig erkannt.  
Die schwierigste Trennung bleibt Inattentive gegen Aggressive, was inhaltlich plausibel ist, weil sich beide Zustände in manchen Dynamikphasen überlappen können.  
Trotzdem bleibt die Gesamtleistung stabil hoch.

---

## Folie 14 - Datensatzstatistik (ca. 40 Sek.)
Die Ergebnisse basieren auf einem großen Datensatz:  
5,037,534 Samples, 37 Features, 80/20 Split mit über 1 Million Testsamples.  
Das ist wichtig, weil die Modellqualität dadurch nicht nur auf kleinen Demo-Daten beruht.

---

## Folie 15 - Top Features & Insights (ca. 55 Sek.)
Bei den wichtigsten Features sehen wir zwei dominante Gruppen:  
erstens Blick-/Aufmerksamkeitsmerkmale wie Off-Road-Gaze und Spurabweichung, zweitens Fahrdynamik wie Jerk und Steering Rate.  
Physiologie unterstützt das Modell zusätzlich, ist aber nicht der stärkste Treiber.  
Das bestätigt die ursprüngliche Hypothese, dass multimodale Fusion sinnvoll ist.

---

## Folie 16 - Zusammenfassung & Ausblick (ca. 45-60 Sek.)
Zusammenfassend wurde eine komplette Pipeline von Rohdaten bis zur Klassifikation umgesetzt.  
Das beste Modell ist LightGBM mit rund 92% Test-Accuracy und hoher Kappa-Übereinstimmung.  
Als nächste Schritte sehe ich vor allem Real-World-Validierung, Echtzeitintegration in Assistenzsysteme und eine Erweiterung um weitere unüberwachte bzw. sequenzbasierte Methoden.  
Vielen Dank.

---

## Q&A Kurzantworten (optional)
### Wie valide sind die Labels ohne Ground Truth?
Die Labels sind Pseudo-Labels und damit eine Limitation. Ich habe versucht, die Validität durch globale Schwellenwerte, Persistenz und session-basierte Evaluation zu stabilisieren.

### Warum LightGBM statt Deep Learning?
Für tabellarische Sensordaten ist LightGBM sehr stark, effizient und leichter reproduzierbar. Deep Learning wäre ein sinnvoller nächster Schritt für zeitliche Sequenzmodellierung.

### Was ist der größte nächste Hebel?
Ein manuell annotiertes Validierungs-Subset aus Realfahrtdaten, um die Labelqualität extern zu prüfen.

