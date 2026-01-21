# Driver State Classification Pipeline - Vollständige Dokumentation

## 1. Problemstellung

Rohsensorik aus Fahrtsimulatoren. **Keine vorhandenen Labels.**

### Ziel
Klassifikation der Fahrerzustände in drei Kategorien:
- **Attentive** (Aufmerksam) - Fahrer konzentriert auf die Straße
- **Inattentive** (Unaufmerksam) - Fahrer abgelenkt (Handy, schläfrig, etc.)
- **Aggressive** (Aggressiv) - Rücksichtsloses/unsicheres Fahren

### Kernproblem
Keine Ground-Truth-Labels vorhanden → **Labels müssen algorithmisch erzeugt werden**

Dies führt zu einer kritischen Frage: **Wie generiert man valide Labels, ohne Circular Logic zu erzeugen?**
(Das Modell soll nicht nur die Labeling-Regeln lernen, sondern echte Fahrerverhalten klassifizieren)

### Warum ist das schwierig?
- Keine manuellen Labels für Trainingsdaten verfügbar
- Vermeidung von "Circular Logic" - das Modell soll nicht nur die Labeling-Regeln lernen
- Sensordatenqualität und Synchronisation mehrerer Sensoren (100 Hz vs 90 Hz vs 1 Hz)
- Unterschiedliche Fahrer und Fahrstile müssen konsistent erkannt werden
- Sensoren können Fehler haben, Daten können fehlende Werte enthalten

## 2. Motivation

- **Fahrersicherheit:** Automatische Zustandsdetektion kann frühzeitig warnen
- **Fahrerassistenzsysteme:** Können proaktiv reagieren bevor kritische Situationen entstehen
- **Versicherungsbranche:** Dokumentation von Fahrerverhalten für Unfallanalysen und -prävention
- **Präventive Maßnahmen:** In modernen Fahrzeugen integrierbar für echtzeitliche Warnungen

## 3. Ziele der Pipeline

1. Vollständige end-to-end Pipeline von Rohsensoren bis zu Predictions
2. Feature-Extraktion aus Fahrzeug-, Eye-Tracking- und Physiologie-Daten
3. Labels mittels **Absolute Thresholds** (keine Circular Logic)
4. Modellvergleich: LightGBM vs. Random Forest
5. Achieve **>90% Accuracy** mit stabiler Generalisierung

---

## 4. Datenerfassung und Verarbeitung

### 4.1 Sensor-Parameter Spezifikation

Die Daten umfassen:
- **Fahrzeugdaten:** Geschwindigkeit, Beschleunigung, Jerk, Steuerwinkel, Headway, Spurabweichung
- **Fahreraufmerksamkeitsdaten:** Blickrichtung (Gaze Heading), Augenlidentöffnung, NDRT-Performance
- **Physiologische Daten:** Herzfrequenz, HRV (SDNN, RMSSD)
- **Szenario-Daten:** Straßensegmente, Verkehrssituationen

### 4.2 Datenbereinigung

```python
def clean_sensor_data(df):
    # Entfernen von NaN-Zeilen
    df = df.dropna(subset=['oveBodyVelocityX', 'steeringWheelAngle'])
    
    # Clipping: Physikalisch unmögliche Werte
    df['oveBodyAccelerationLongitudinalX'] = \
        df['oveBodyAccelerationLongitudinalX'].clip(-15, 15)
    
    # Outlier Detection (IQR)
    Q1 = df['steering_rate'].quantile(0.25)
    Q3 = df['steering_rate'].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df['steering_rate'] >= Q1 - 1.5*IQR) & \
           (df['steering_rate'] <= Q3 + 1.5*IQR)
    df = df[mask]
    
    return df
```

Interpolation fehlender Werte
•	Lineare oder spline-basierte Interpolation
•	Bewertung der Interpolationsgüte mittels Fehlerkennwerten

Synchronisation der Sensorströme
Zeitliche Abstimmung von:
•	Eye-Tracking-Daten
•	Fahrzeugdynamik
•	Physiologie / MiRA-Zonen
Verfahren: Timestamp-Matching, Resampling, Merge

Datenqualitätsprüfung
•	Korrelation zwischen Fahrzeugdaten
•	Korrelation zwischen Aufmerksamkeitsmerkmalen
•	Intermodale Korrelation (Fahrzeug  Fahrer)
Deliverables:
Ergebnis	Beschreibung
Bereinigter & synchronisierter Datensatz	Grundlage für Modelltraining
Visualisierung von Gaze-, TTC- & Fahrzeugsignalen	Zeitbasierte Analyse
Bericht zur Datenqualität	inkl. Interpolationsanalyse

eature Engineering & Explorative Analyse
Statistische & dynamische Merkmale
•	Mittelwert, Standardabweichung, Varianz
•	Dynamische Merkmale: Beschleunigungsgradienten, Lenkdynamik, Gaze-Shift-Frequenz
Verhaltensmerkmale
Fahrertyp	Parameterbeispiele
Unaufmerksam	lange Blickabwendung, hoher TTC, verzögertes Bremsen
Aggressiv	hohe Beschleunigungsspitzen, geringer Headway, hohe Lenkdynamik
Kausalanalyse
•	Zusammenhang Aufmerksamkeit ↔ Fahrzeugreaktion
•	Signifikante Einflussfaktoren identifizieren
Visual Output:
•	Heatmaps
•	Korrelationsmatrizen
•	Verteilungsdiagramme
Deliverables:
•	Feature-Matrix
•	Explorative Diagramme
•	Bericht: wichtigste Einflussgrößen
________________________________________
7. Modellierung
Modelltyp: Supervised Learning
•	Klassifikation: Fahrerzustand (Attentive, Inattentive, Aggressive)
•	Modelle:
• LightGBM (Gradient Boosting, 200 Bäume, Leaf-wise Wachstum, learning rate 0.05)
• Random Forest (Baseline)
Trainingsprozess:
•	Leave-One-Session-Out Cross-Validation
•	Training auf N-1 Sessions, Test auf 1 Session
•	Evaluation: Accuracy, F1, Cohen’s Kappa
•	Finale Modelltraining auf allen Daten
Kostenfunktion:
•	Cross-Entropy für Klassifikation
________________________________________
8. Evaluation
Verwendete Metriken:
•	Accuracy (91,13 % ± 2,62 % für LightGBM, RF: 87,44 % ± 3,71 %)
•	Precision, Recall, F1-Score (pro Klasse)
•	Cohen’s Kappa (0,84 – nahezu perfekte Übereinstimmung)
•	Standardabweichung über Sessions
Erkenntnisse:
•	LightGBM stabiler und genauer als Random Forest
•	Höchste Trennschärfe bei schwer unterscheidbaren Klassen (Inattentive vs Aggressive)
________________________________________
9. Modelloptimierung und Validierung
•	Hyperparameter-Tuning für LightGBM
•	Feature Importance Analyse (z. B. SHAP)
•	Validierung auf realen Daten (falls verfügbar)
•	Erweiterung durch Ensemble-Ansätze (XGBoost)
________________________________________
10. Schlussfolgerung
•	Vollständige Pipeline von Rohdaten bis Predictions implementiert
•	35 Features aus Fahrzeug-, Eye-Tracking- & Physiologie-Daten
•	Labels algorithmisch via Absolute Thresholds & Accumulative Scores erzeugt
•	LightGBM liefert beste Accuracy & Stabilität
•	Pipeline produktionsbereit, realistische Generalisierung über Sessions
________________________________________
11. Literaturverzeichnis
(Platzhalter für wissenschaftliche Quellen, APA oder andere Normen)
________________________________________
12. Anhang
•	Code & Scripts für Pipeline
•	Feature-Definitionen
•	Diagramme: Verteilungen, Heatmaps, Korrelationen
•	Datenbeschreibung & Beispieltabellen










