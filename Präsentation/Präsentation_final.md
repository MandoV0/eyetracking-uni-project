# Präsentation: Fahrerzustand-Klassifikation (Final)

Geplante Gesamtdauer: 15:00 Minuten  
Sprechplan: 13:30 Vortrag + 1:30 Q&A  
Zielgruppe: Betreuer/Prüfungskommission Informatik

---

## Zeitplan pro Folie
1. Folie 1: 0:25  
2. Folie 2: 0:55  
3. Folie 3: 0:45  
4. Folie 4: 0:50  
5. Folie 5: 0:50  
6. Folie 6: 1:15  
7. Folie 7: 0:35  
8. Folie 8: 0:40  
9. Folie 9: 0:50  
10. Folie 10: 0:40  
11. Folie 11: 0:50  
12. Folie 12: 1:00  
13. Folie 13: 0:55  
14. Folie 14: 0:40  
15. Folie 15: 0:55  
16. Folie 16: 1:25

Summe Vortrag: 13:30  
Geplante Q&A-Zeit: 1:30  
Gesamtslot: 15:00

---

## Folie 1 - Titel
Zeit: 0:25  
Kernaussage:
- Thema, Klassen und Zielbild knapp setzen

---

## Folie 2 - Motivation & Problemstellung
Zeit: 0:55  
Inhalt:
- Sicherheitsrelevanz
- Grenzen bestehender DMS
- Keine Ground-Truth-Labels

Kernaussage:
- Kernproblem ist Labeling ohne zirkuläre Logik

---

## Folie 3 - Ziele der Arbeit
Zeit: 0:45  
Inhalt:
- Absolute Threshold Labeling
- Temporal Persistence
- 37 multimodale Features
- End-to-End Pipeline

Kernaussage:
- Methodischer und technischer Gesamtbeitrag

---

## Folie 4 - Datenerfassung & Modalitäten
Zeit: 0:50  
Inhalt:
- Fahrzeugdynamik
- Eye-Tracking
- Physiologie
- Umfelddaten

Kernaussage:
- Multimodale Fusion statt Single-Sensor

---

## Folie 5 - Feature Engineering
Zeit: 0:50  
Inhalt:
- Aggressive- und Inattentive-Indikatoren
- Kontext- und Physio-Features
- Rolling-Window-Ansatz

Kernaussage:
- Features bilden die Klassen semantisch ab

---

## Folie 6 - Label-Generierung
Zeit: 1:15  
Inhalt:
- Data-driven Thresholds
- Kritische Events
- Akkumulative Scores
- 2s Persistenz

Kernaussage:
- Stabilere Pseudo-Labels trotz fehlender GT

---

## Folie 7 - Label-Verteilung
Zeit: 0:35  
Inhalt:
- Klassendistribution
- Imbalance-Einordnung

Kernaussage:
- Balancing im Modell explizit adressiert

---

## Folie 8 - Modellierung: Random Forest
Zeit: 0:40  
Inhalt:
- RF als Baseline
- zentrale Hyperparameter

Kernaussage:
- Vergleichsanker für LightGBM

---

## Folie 9 - Random Forest Ergebnisse
Zeit: 0:50  
Inhalt:
- Accuracy, Kappa, F1 je Klasse

Kernaussage:
- solide Baseline, aber unter LGBM

---

## Folie 10 - RF Ergebnis-Einordnung
Zeit: 0:40  
Inhalt:
- Fehlerbilder und Konfusionen
- Grenzen bei subtilen Mustern

Kernaussage:
- Bedarf für stärkeres Boosting-Modell

---

## Folie 11 - Modellierung: LightGBM
Zeit: 0:50  
Inhalt:
- Hyperparameter
- Boosting-Prinzip
- GPU-Unterstützung

Kernaussage:
- bessere Generalisierung und Effizienz

---

## Folie 12 - LightGBM Ergebnisse & Vergleich
Zeit: 1:00  
Inhalt:
- Train/Test Accuracy
- Kappa
- Per-Class F1

Kernaussage:
- klar bestes Modell im Projekt

---

## Folie 13 - Confusion Matrix
Zeit: 0:55  
Inhalt:
- klassenspezifische Fehleranalyse

Kernaussage:
- Attentive sehr stabil, Grenzfälle bei Inattentive/Aggressive

---

## Folie 14 - Datensatzstatistik
Zeit: 0:40  
Inhalt:
- 5,037,534 Samples
- 37 Features
- 80/20 Session-Split

Kernaussage:
- Ergebnis auf großer Datenbasis

---

## Folie 15 - Top Features & Insights
Zeit: 0:55  
Inhalt:
- wichtigste Treiber je Klasse

Kernaussage:
- Blickverhalten + Fahrdynamik dominieren

---

## Folie 16 - Zusammenfassung & Ausblick
Zeit: 1:25  
Inhalt:
- Kernbeiträge
- Limitationen
- nächste Schritte

Kernaussage:
- robuste Pipeline, klare Roadmap für Real-World-Validierung

---

## Q&A-Hinweis
Separater Q&A-Katalog: `QA_final.md`
