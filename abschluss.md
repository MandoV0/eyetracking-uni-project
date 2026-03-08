Titelseite – Standardformat 
Titel der Arbeit
Untertitel (falls vorhanden)

Vorname Nachname des Studenten
Matrikelnummer: XXXXXXX

Betreut von:
Titel Vorname Nachname
(Institut / Lehrstuhl)

Fakultät für Informatik
Universitätsname
Studiengang (z. B. Master Informatik)


Ort, Monat Jahr












Leibniz Universität Hannover
Fakultät für Elektrotechnik und Informatik
Institut für … 



Analyse und Klassifikation von Fahreraufmerksamkeits- und Fahrverhaltensmustern mittels Deep Learning




Informatik Projektarbeit
eingereicht von:
Dein Name
Matrikelnummer: XXXXXXX




Betreuer:
Prof. Dr. rer. nat. Jamal Raiyn 




Hannover, Februar 2025
INHALTSVERZEICHNIS
1.	Einführung
2.1 State of the Art
2.2 Problemstellung
2.3 Motivation
2.4 Ziele
2.	Daten und Methodik
3.1 Datenerfassung
3.2 Parameter-Spezifikation
3.3 Reinigung fehlerhafter Daten
3.4 Interpolation fehlender Werte
3.5 Synchronisation der Sensorströme
3.6 Datenqualitätsanalyse
3.	Feature Engineering & Explorative Analyse
4.1 Statistische Features
4.2 Dynamische Features
4.3 Klassifikation von Fahrertypen
4.4 Kausalitätsanalyse
4.5 Visualisierung der Merkmalsverteilungen
4.	Modellierung
5.1 Deep Learning Architektur
5.2 Klassifikationsmodell
5.3 Vorhersagemodell
5.4 Trainings- und Testdaten
5.5 Hyperparameter & Kostenfunktion
5.	Evaluation
6.1 Accuracy
6.2 Precision, Recall, F1
6.3 Korrelation & statistische Kennwerte
6.4 MSE & RMSE
6.5 Analyse der Modellgewichte
6.	Modelloptimierung & Validierung
7.1 Tests mitCTAG
7.2 Tests mit Videodaten (freiswilllig)
7.3 Vergleich mit anderen Modellen (nennen sie das model)
7.4 Szenarien durch generative KI/ model optimierung
7.	Schlussfolgerung
8.	Literaturverzeichnis


9.	Anhang
10.1	Abreitspakete
10.2	Code
10.3	Datenspezifikation
10.4	Weite Diagramme


















Projektplanung in Arbeitspakete

Sprint 1 (Woche 1–3 bis  29.Okt. 2025): Grundlagen & Datenpipeline
Ziel: Aufbau der technischen Umgebung und grundlegenden Datenerfassung.

Arbeitspakete:
1.	Umgebungsaufbau
o	Einrichtung der Python-Umgebung, Git-Repository, Ordnerstruktur
o	Installation von Abhängigkeiten (pandas, numpy, numba, scipy, matplotlib)
2.	Szenario-Struktur anlegen
o	Definition der Simulationsumgebung (Ego-, Lead-, Cut-In-Fahrzeug)
o	Erste Implementierung von cut_in, cut_out, car_following
3.	Sensorik & Datenpipeline
o	Aufbau von Datenfluss: Input (Parameter) → Simulation → Output (CSV)
o	Synchronisation von Zeitreihen (Position, Geschwindigkeit, Beschleunigung)

Sprint 2 (Woche 3–3  bis zum 05.11.2025): Datenaufbereitung
Ziel: Datenqualität sichern und Analysefähigkeit herstellen.
Arbeitspakete:
1.	Filterung & Interpolation
o	Glätten von Trajektorien und Geschwindigkeitssignalen
o	Zeitliche Interpolation (z. B. 0.1 s Schrittweite)
2.	Normalisierung & Einheitlichkeit
o	Umrechnung in einheitliche Maßeinheiten (m/s, m, s)
3.	Datenbank/Strukturaufbau
o	Szenariodaten in strukturierter Form speichern (JSON oder SQLite)
o	Optional: Cloud-Speicherung (GitHub Repo oder lokal)
4.	Parameter specification and corrolation

Sprint 3 (Woche 4–5 bis zum 19.11.2025): Feature Engineering
Ziel: Definition und Berechnung relevanter Merkmale für die Modellbildung.
Arbeitspakete:
1.	Merkmalsdefinition
o	Dynamische Parameter: Abstand, Relativgeschwindigkeit, TTC, Beschleunigung
o	Kontextparameter: Verkehrsdichte, Spurwechselstatus
2.	Feature-Berechnung
o	Python-Modul zur automatischen Feature-Berechnung pro Szenario
o	Speicherung in strukturierter Form
3.	Validierung der Features
o	Visualisierung von Feature-Verläufen
o	Prüfung auf physikalische Plausibilität
Deliverables:
•	vollständiges Feature-Set
•	Validierungsplots (Abstand, Geschwindigkeit, TTC)
•	Python-Modul feature_extractor.py



Sprint 4 (Woche 7–7 bis zum 03.12.2025): Sicherheitsmodelle I – Basismodellierung
Ziel: Implementierung der Kernlogik der Sicherheitsmodelle.
Arbeitspakete:
1.	Implementierung FSM (Fuzzy Safety Model)
o	Definition von Fuzzy-Regeln (z. B. Abstand, Geschwindigkeit, Komfort)
o	Berechnung von „Criticality Index“ pro Zeitschritt
2.	Implementierung RSS
o	Berechnung der sicheren Längs- und Querabstände nach RSS-Formeln
o	Implementierung der Reaktionslogik bei Gefährdung
3.	Modell-Schnittstelle
o	Einheitliches Interface für alle Modelle (Input: Szenariodaten, Output: Sicherheitsbewertung)
Deliverables:
•	funktionsfähige FSM- und RSS-Modelle
•	Vergleich der Modelle im one_case-Modus
•	erste Sicherheitsmetriken (z. B. min TTC, safe distance)

 Sprint 5 (Woche 8–10 bis zum 24.12. 2025): Sicherheitsmodelle II – Integration & Erweiterung (eigenes Model)
Ziel: Integration weiterer Modelle und Vereinheitlichung der Architektur.
Arbeitspakete:
1.	Implementierung Reg157
o	Umsetzung der UNECE-Regelmechanismen für Spurhaltung & Abstand
o	Validierung gegen Beispielparameter
2.	Implementierung CC_human_driver
o	Modellierung menschlichen Fahrverhaltens (Reaktionszeit, Komfortgrenzen)
3.	Integration & Vergleich
o	Gemeinsame API für alle Modelle
o	Visualisierung der Unterschiede (z. B. Abstandsverlauf, Reaktionszeit)
Deliverables:
•	vollständige Model Library (models/)
•	konsistente Schnittstelle für Simulation
•	Vergleichsgrafiken aller Modelle im cut-in-Szenario





Sprint 6 (Woche 11–12 bis zum 07.01.2026): Simulation & Vergleichsanalyse
Ziel: Systematische Untersuchung der Modelle über verschiedene Szenarien.
Arbeitspakete:
1.	„Comparison “-Modus
o	Batch-Ausführung mehrerer Szenarien für alle Modelle
o	Speicherung von TTC-, Abstand- und Bremswerten
2.	Datenanalyse
o	Statistische Auswertung (Mittelwerte, Standardabweichung, Häufigkeiten)
3.	Visualisierung
o	TTC-Zeitverlauf, Heatmaps, Boxplots pro Modelltyp
Deliverables:
•	Simulationsdatenbank mit Vergleichsergebnissen
•	Analyseplots (RSS vs FSM vs Reg157)
•	Bericht „Modellvergleich kritischer Szenarien“

 Sprint 7 (Woche 13–14 bsi zum 21.01.2025): Evaluation & Validierung
Ziel: Bewertung der Modelle hinsichtlich Sicherheit, Robustheit und Komfort. 
Arbeitspakete:
1.	Evaluationsmetriken
o	TTC-Minimum, Anzahl Kollisionen, Komfortindex
o	Vergleich menschlicher vs. formaler Modelle
2.	Validierung mit realistischen Parametern
o	Anpassung von Längs-/Querbeschleunigungen, Verzögerungen
3.	Berichtserstellung (technische Evaluation)
o	Interpretation der Ergebnisse
o	Bewertung der Modelle im Kontext UNECE R157
Deliverables:
•	evaluierte Systemversion
•	Tabellen & Grafiken mit Performance-Vergleich
•	technischer Evaluationsbericht



 Sprint 8 (Woche 15–16 bis zum 04.02.2026): Abschlussphase
Ziel: Finalisierung, Dokumentation, Präsentation.
Arbeitspakete:
1.	Systemvalidierung
o	End-to-End-Test aller Funktionen
o	Stabilitätsprüfung (mehrere Runs)
2.	Dokumentation
o	Code-Kommentare, technische Beschreibung, Setup-Anleitung
o	wissenschaftlicher Abschlussbericht (z. B. 15–20 Seiten)
3.	Präsentation
o	PowerPoint mit Ergebnissen, Grafiken, Empfehlungen
Deliverables:
•	validierte Endversion der Bibliothek
•	Dokumentation (PDF oder Word)
•	Abschlusspräsentation








 
