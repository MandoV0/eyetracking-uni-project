Fahrerverhaltensklassifizierung mittels Random Forest und anomaliebasiertem HMM

1. State of the Art
Aktuelle Ansätze zur Fahrerverhaltenserkennung nutzen häufig maschinelles Lernen, um aus Fahrzeug- und Blickdaten auf den Zustand des Fahrers zu schließen. Während klassische Ansätze oft auf festen Schwellwerten basieren, erlauben moderne Verfahren wie Random Forests und Hidden Markov Modelle (HMM) eine robustere Klassifizierung unter Berücksichtigung zeitlicher Abhängigkeiten.

________________________________________
2. Problemstellung
Die Identifikation von unaufmerksamem oder aggressivem Fahrverhalten ist entscheidend für die Sicherheit im Straßenverkehr. Die Herausforderung besteht darin, aus einer Vielzahl heterogener Sensordaten (Fahrzeugdynamik, Physiologie, Blickverhalten) präzise und zeitlich konsistente Aussagen über den Fahrerzustand zu treffen.

________________________________________
3. Motivation
Durch die frühzeitige Erkennung kritischer Zustände können Fahrerassistenzsysteme (ADAS) gezielter eingreifen oder warnen. Ein hybrider Ansatz aus Anomalieerkennung und Klassifizierung ermöglicht es, auch seltene, aber sicherheitskritische Ereignisse zuverlässig zu erfassen.

________________________________________
4. Ziele
•	Entwicklung eines robusten Datenverarbeitungsprotokolls zur Synchronisation von Ego-, Physiologie- und Umfelddaten.
•	Implementierung eines hybriden Klassifizierungsmodells (RF-HMM-Anomaly).
•	Evaluierung der Modellperformance hinsichtlich Genauigkeit und zeitlicher Stabilität.

________________________________________
5. Datenerfassung und -verarbeitung
5.1 Parameter-Spezifikation
Die Daten umfassen:
•	Fahrzeugdaten: Längs- und Querbeschleunigung, Geschwindigkeit, Ruck (Jerk), Gaspedalstellung, Lenkwinkel, Gierrate, Fahrspurposition.
•	Fahreraufmerksamkeitsdaten: Gaze-Heading/-Pitch (Blickrichtung), Augenlidöffnung.
•	Physiologiedaten: EKG-Rohdaten zur Berechnung von Herzfrequenz (HR) und RR-Intervallen (HRV).
•	Szenario-Daten: Anzahl der Umfeldobjekte, minimaler Abstand zu anderen Fahrzeugen (surround_min_dist), Zeitlücke (THW).

5.2 Datenbereinigung
•	Outlier Detection: Identifikation extremer Messwerte in der Beschleunigung und im EKG.
•	Rauschunterdrückung: Glättung der Signale mittels rollierender Fenster (Rolling Windows).

5.3 Interpolation fehlender Werte
•	Lineare und splinebasierte Interpolation (Cubic Spline) zur Angleichung unterschiedlicher Abtastraten der Sensoren (z. B. EKG vs. Fahrzeugbus).

5.4 Synchronisation der Sensorströme
•	Verwendung von `merge_asof` für Nearest-Neighbor-Matching der Timestamps.
•	Resampling aller Datenströme auf die Zeitbasis des Ego-Fahrzeugs.

5.5 Datenqualitätsprüfung
•	Berechnung der Korrelation zwischen physiologischen Stressindikatoren (HRV) und aggressiven Fahrmanövern (starker Ruck).
•	Überprüfung der Konsistenz von Blickdaten und Spurhalteverhalten.

Deliverables
Ergebnis	Beschreibung
Bereinigter & synchronisierter Datensatz	Fusionierte Feature-Matrix aus Ego-, Physio- und Umfelddaten.
Visualisierung von Gaze-, TTC- & Fahrzeugsignalen	Zeitbasierte Diagramme zur Analyse von Aufmerksamkeitsdefiziten.
Bericht zur Datenqualität	Analyse der Interpolationsgüte und Synchronisationsgenauigkeit.

________________________________________
6. Feature Engineering & Explorative Analyse
Statistische und dynamische Merkmale
•	Aggregierte Merkmale über Zeitfenster (Rolling Windows): Mittelwert, Standardabweichung und Maxima von Beschleunigung und Ruck.
•	Steuerungsdynamik: Lenkwinkelrate und Frequenz der Gaspedaländerungen.
•	Physiologische Features: SDNN und RMSSD der RR-Intervalle zur Stressanalyse.

Verhaltensmerkmale
Fahrertyp	Parameterbeispiele
Unaufmerksam	Hohe Varianz der Querposition, hohe Blickabwendung (Off-Road Ratio), niedrige Augenlidöffnung, hohe Fehlerquote bei Nebenaufgaben (NDRT).
Aggressiv	Hohe Beschleunigungsmaxima, hohe Jerk-Werte, geringer Zeitabstand (Short Headway), hohe Herzfrequenz.
Aufmerksam	Hohe "Control Smoothness", stabile Spurhaltung, Blick primär auf der Fahrbahn.

Kausalanalyse
•	Untersuchung des Zusammenhangs zwischen physiologischem Stress (HRV sinkt) und aggressivem Lenkverhalten.
•	Identifikation der Blickrichtung als primärer Prädiktor für Unaufmerksamkeit.

Visual Output:
•	Heatmaps der Feature-Korrelationen.
•	Feature-Importance-Plots des Random Forests.
•	Konfusionsmatrizen zur Klassifikationsbewertung.

Deliverables
Feature-Matrix	Matrix mit berechneten rollierenden Statistiken.
Explorative Diagramme	Visualisierung der Verteilung von Aggressions- und Inattentions-Scores.
Bericht: Einflussgrößen	Ranking der wichtigsten Features (z. B. Jerk, Gaze-Ratio).

________________________________________
7. Modellierung
Modelltyp: RF-HMM-Anomaly Hybrid
1.	Anomalieerkennung (Isolation Forest): Identifikation von Abweichungen vom normalen Fahrverhalten (ca. 15-20% der Daten).
2.	Clustering (K-Means): Einteilung der Anomalien in "Aggressiv" und "Unaufmerksam" basierend auf Feature-Profilen.
3.	Temporal Smoothing (Hidden Markov Modell): Anwendung eines GaussianHMM zur Glättung der Zustandsübergänge und Vermeidung von Rauschen in der Klassifizierung.
4.	Klassifizierung (Random Forest): Finales Modell zur Vorhersage der drei Klassen (Aufmerksam, Unaufmerksam, Aggressiv).

Trainingsprozess
•	80/20 Train-Test-Split unter Berücksichtigung einer zeitlichen Lücke (Gap), um Datenlecks durch rollierende Fenster zu vermeiden.
•	Balancierung der Klassen durch `class_weight='balanced'`.
•	Optimierung über 150 Estimators und Begrenzung der Baumtiefe zur Vermeidung von Overfitting.

Kostenfunktion
•	Log Loss (Cross-Entropy) für die Wahrscheinlichkeitsbewertung der Klassifikation.

________________________________________
8. Evaluation
Verwendete Metriken:
•	Accuracy: Gesamtgenauigkeit des Modells (Test-Accuracy ca. 85%).
•	Precision/Recall/F1-Score: Speziell für die Minderheitsklassen (Aggressiv/Unaufmerksam).
•	Cohen's Kappa & Matthews Correlation Coefficient: Bewertung der Übereinstimmung jenseits des Zufalls.
•	Log Loss: Bewertung der Vorhersageunsicherheit.
•	Mean Squared Error (MSE): Für die Validierung der zugrundeliegenden Scores.

________________________________________
9. Modelloptimierung und Validierung
•	Validierung mittels 5-Fold Cross-Validation.
•	Testen auf unterschiedlichen Szenarien (VTI-Daten).
•	Analyse der Modellstabilität durch Vergleich von initialen heuristischen Labels mit HMM-geglätteten Labels.

________________________________________
10. Schlussfolgerung
Das hybride Modell zeigt eine hohe Genauigkeit bei der Unterscheidung von Fahrzuständen. Besonders die Kombination aus Anomalieerkennung und zeitlicher Glättung mittels HMM führt zu einer realistischeren Abbildung des Fahrerverhaltens, da sprunghafte Zustandswechsel minimiert werden. Zukünftige Arbeiten sollten die Generalisierbarkeit auf unbekannte Probanden weiter untersuchen.

________________________________________
11. Literaturverzeichnis
•	Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
•	Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.

________________________________________
12. Anhang
•	Code-Auszüge: `AnomalyBasedClassifier`, `DataProcessor`.
•	Diagramme: Feature Importance (Top-Features: Jerk, Steering Rate, Gaze-Ratio).
•	Datenbeschreibung: Spezifikation der VTI/i4driving Datensätze.