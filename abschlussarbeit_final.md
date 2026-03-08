# Leibniz Universität Hannover
Fakultät für Elektrotechnik und Informatik  
Institut für **[bitte ergänzen]**

## Analyse und Klassifikation von Fahrerzuständen aus Eye-Tracking- und Fahrsimulator-Daten mittels Machine Learning
### Informatik-Projektarbeit (Modulumfang ca. 15 CP)

eingereicht von:  
**Burak Güldüz**  
Matrikelnummer: **1526703**

Betreuer:  
**Prof. Dr. rer. nat. Jamal Raiyn**

Hannover, März 2026

---

## Kurzfassung
Diese Arbeit entwickelt eine durchgängige Pipeline zur Fahrerzustandsklassifikation mit den Klassen **Attentive**, **Inattentive** und **Aggressive** auf Basis von multimodalen Simulator-Daten (Fahrzeugdynamik, Eye-Tracking, Physiologie und Umfelddaten). Da keine Ground-Truth-Labels vorliegen, wird ein regelbasiertes Labeling mit globalen, datengetriebenen Schwellenwerten und zeitlicher Persistenz eingesetzt. Darauf aufbauend werden mehrere Modelle verglichen; **LightGBM** liefert die beste Gesamtleistung.

Für einen repräsentativen Session-basierten Split (80/20) auf ca. **5.04 Mio. Zeitpunkten** mit **37 Features** erreicht LightGBM **91.97% Test-Accuracy** bei **Cohen’s Kappa = 0.8698** und übertrifft den Random-Forest-Baseline-Ansatz deutlich. Zusätzlich wurde ein unüberwachter GMM-Vergleich durchgeführt, der erwartungsgemäß hinter dem überwachten Ansatz liegt. Die Ergebnisse zeigen, dass der gewählte Pipeline-Aufbau trotz fehlender manueller Labels eine robuste und praktisch nutzbare Klassifikation ermöglicht.

---

## Inhaltsverzeichnis
1. Einführung  
1.1 State of the Art  
1.2 Problemstellung  
1.3 Motivation  
1.4 Ziele und Beiträge  
2. Daten und Methodik  
2.1 Datenerfassung  
2.2 Parameter-Spezifikation  
2.3 Reinigung fehlerhafter Daten  
2.4 Interpolation fehlender Werte  
2.5 Synchronisation der Sensorströme  
2.6 Datenqualitätsanalyse  
3. Feature Engineering und explorative Analyse  
3.1 Statistische Features  
3.2 Dynamische Features  
3.3 Klassennahe Verhaltensindikatoren  
3.4 Korrelationsanalyse  
3.5 Visualisierung der Merkmalsverteilungen  
4. Modellierung  
4.1 Modellarchitektur der Pipeline  
4.2 Labelgenerierung durch Absolute Thresholds  
4.3 Klassifikationsmodell (LightGBM)  
4.4 Vergleichsmodelle (Random Forest, GMM)  
4.5 Trainings- und Testdaten  
4.6 Hyperparameter und Kostenfunktion  
5. Evaluation  
5.1 Accuracy  
5.2 Precision, Recall, F1-Score  
5.3 Konfusionsmatrix und Cohen’s Kappa  
5.4 Einordnung von MSE/RMSE im Klassifikationskontext  
5.5 Analyse der Modellgewichte (Feature Importance)  
6. Modelloptimierung und Validierung  
6.1 Session-übergreifende Validierung  
6.2 Laufzeit- und GPU-Analyse  
6.3 Vergleich mit anderen Modellen  
6.4 Szenarien, Robustheit und Einsatz von generativer KI  
7. Schlussfolgerung  
8. Literaturverzeichnis  
9. Anhang  
9.1 Arbeitspakete  
9.2 Codeübersicht  
9.3 Datenspezifikation  
9.4 Weitere Diagramme

---

## 1. Einführung

### 1.1 State of the Art
Driver-Monitoring-Systeme (DMS) sind ein zentraler Baustein moderner Fahrerassistenz. Klassische Systeme nutzen einzelne Signale wie Lenkmomente oder einfache Heuristiken (z. B. Blickabwendung über x Sekunden). Aktuelle Forschung zeigt, dass robuste Zustandsdiagnostik meist eine **multimodale Fusion** benötigt: Fahrdynamik, Blickverhalten und physiologische Indikatoren ergänzen sich.

Für tabellarische Sensordaten haben sich Gradient-Boosting-Verfahren (u. a. LightGBM) als sehr leistungsfähig erwiesen. Ein Kernproblem in Simulatorstudien bleibt jedoch bestehen: Es fehlen häufig direkte Ground-Truth-Labels für Zustände wie Aufmerksamkeit oder Aggressivität.

### 1.2 Problemstellung
Im vorliegenden Projekt sollen drei Fahrerzustände klassifiziert werden:
- **Attentive (0)**: stabile, aufmerksame Fahrweise
- **Inattentive (1)**: abgelenkte oder reduzierte Aufmerksamkeit
- **Aggressive (2)**: risikoreiche, ruckartige Fahrweise

Die zentrale Herausforderung ist die fehlende Labelbasis. Damit stellt sich die Frage, wie aus unlabeled Daten sinnvolle Trainingsziele erzeugt werden können, ohne nur die Heuristik zu reproduzieren.

### 1.3 Motivation
Die praktische Relevanz ergibt sich aus:
- Früherkennung kritischer Zustände für Assistenzsysteme
- Verbesserter Sicherheit bei Teilautomatisierung
- Nachvollziehbarer, datenbasierter Bewertung von Fahrverhalten
- Übertragbarkeit der Pipeline auf weitere Simulator- und Realfahrtdaten

### 1.4 Ziele und Beiträge
Die Arbeit verfolgt vier Hauptziele:
- Aufbau einer robusten End-to-End-Pipeline von Rohdaten bis Modellbewertung
- Entwicklung eines konsistenten Labeling-Ansatzes ohne manuelle Annotation
- Vergleich mehrerer Modelle mit Fokus auf Generalisierungsfähigkeit
- Technische Dokumentation der wichtigsten Designentscheidungen

Konkrete Beiträge:
- Session-basierte Datenaufbereitung und Splits zur Leak-Vermeidung
- 37 engineered Features aus multimodalen Signalen
- Absolute-Threshold-Labeling mit temporaler Persistenz
- Quantitativer Vergleich: LightGBM vs. Random Forest (+ explorativ GMM)

---

## 2. Daten und Methodik

### 2.1 Datenerfassung
Die Datengrundlage stammt aus dem VTI-Kontext (`data/VTI/T3.2`) mit mehreren synchronisierbaren Quellen pro Session:
- Ego-Fahrzeugdaten
- Physiologie (ECG)
- Surrounding-Vehicle-Daten
- Eye-Tracking-Merkmale in den Ego-Dateien

Die Pipeline koppelt zusammengehörige Dateien über eine gemeinsame ID pro Session.

### 2.2 Parameter-Spezifikation
Verwendete Signalgruppen (Auszug):

| Modalität | Beispielspalten | Bedeutung |
|---|---|---|
| Fahrdynamik | `oveBodyAccelerationLongitudinalX`, `oveBodyJerkLongitudinalX`, `steeringWheelAngle`, `throttle`, `aheadTHW` | Dynamik, Kontrollverhalten, Abstandsverhalten |
| Spurverhalten | `ovePositionLateralR`, `oveYawVelocity` | Spurstabilität und Richtungsänderung |
| Eye-Tracking | `openxrGazeHeading`, `openxrGazePitch`, `varjoEyelidOpening` | Blickabwendung, Müdigkeitsindikatoren |
| NDRT-Leistung | `arrowsCorrectCount`, `arrowsWrongCount`, `arrowsTimeoutCount` | Ablenkungsnahe Sekundäraufgabe |
| Physiologie | `ecg` (abgeleitet zu `heartRate`, `rrInterval`) | Stress/Belastungsindikatoren |
| Umfeld | aggregierte Distanz- und Dichtewerte | Verkehrskontext |

### 2.3 Reinigung fehlerhafter Daten
Wesentliche Bereinigungsschritte:
- Konsistente Datentypkonvertierung und Zeit-Sortierung
- Behandlung von Platzhalterwerten (z. B. `aheadTHW = -1`)
- Ausschluss offensichtlicher Artefakte in Umfelddistanzen
- Nachbearbeitung physiologischer Ableitungen (Forward/Backward Fill)

### 2.4 Interpolation fehlender Werte
Unterstützte Verfahren:
- **MERGE (0)**: nearest `merge_asof`
- **LINEAR (1)**: lineare Interpolation via `interp1d`
- **SPLINE (2)**: kubische Interpolation

Standard in den finalen Läufen: **LINEAR (1)**.

### 2.5 Synchronisation der Sensorströme
Die Ego-Zeitachse dient als Referenz. Physiologie und Umfelddaten werden darauf abgebildet. Für Surrounding-Daten werden zunächst pro Zeitpunkt Actor-Features aggregiert:
- `surround_actor_count`
- `surround_min_dist`
- `surround_avg_speed`

Dadurch entsteht je Zeitstempel ein konsistenter multimodaler Merkmalsvektor.

### 2.6 Datenqualitätsanalyse
Die Datenqualität wurde über Verteilungsplots, Zeitreihenplots und Korrelationsdarstellungen geprüft (`LightGBMPipeline/plots`). Sichtbare Effekte:
- Plausible Sensorverteilungen nach Bereinigung
- seltene, aber klare Extremwerte in Jerk/Lenkdynamik
- unterscheidbare Muster bei Blickabwendung und Spurstabilität

---

## 3. Feature Engineering und explorative Analyse

### 3.1 Statistische Features
Das Feature Engineering nutzt Rolling-Window-Statistiken (Standard: `window_size=50`). Beispiele:
- Mittelwert, Standardabweichung, Maximum von Beschleunigung/Jerk
- Varianzmaße für Lenkung und Spurlage
- gleitende NDRT-Fehlermaße
- physiologische Fenstermerkmale (`hr_mean`, `hrv_sdnn`, `hrv_rmssd`)

### 3.2 Dynamische Features
Zusätzlich wurden dynamiknahe Größen verwendet:
- `steering_rate` als Änderungsmaß des Lenkwinkels
- `throttle_changes` als Unruheindikator
- `short_headway_ratio` als Anteil kritischer Abstände
- `control_smoothness` als inverse Kontrollunruhe

### 3.3 Klassennahe Verhaltensindikatoren
Typische Muster:
- **Aggressive**: hohe Jerk-/Beschleunigungsdynamik, starkes Lenken, kurze Headways
- **Inattentive**: hohe `gaze_off_road_ratio`, auffällige Spurabweichung, NDRT-Fehler
- **Attentive**: geringe Dynamikextreme, stabilere Spur- und Blickführung

### 3.4 Korrelationsanalyse
Die explorative Korrelationsanalyse zeigt:
- in sich zusammenhängende Aggressionsmerkmalsblöcke (Jerk, Steering, Throttle)
- separate Inattention-Cluster (Gaze, Lateral-Drift, NDRT)
- physiologische Features mit zusätzlichem, aber geringerem Erklärbeitrag

### 3.5 Visualisierung der Merkmalsverteilungen
Erzeugte Diagramme:
- `label_distribution.png`
- `feature_distributions.png`
- `feature_correlation.png`
- `confusion_matrix.png`
- `feature_importance.png`
- `gmm_clusters_visualization.png`

Diese Artefakte wurden für die Ergebnisinterpretation und Diskussion verwendet.

---

## 4. Modellierung

### 4.1 Modellarchitektur der Pipeline
Pipelineablauf:
1. Laden/Synchronisieren der Sessiondaten
2. Feature Engineering (37 Features)
3. Labelgenerierung (Absolute Thresholds + Persistenz)
4. Session-basierter Split oder Leave-One-Session-Out
5. Training und Evaluation (LightGBM, RF, GMM)

### 4.2 Labelgenerierung durch Absolute Thresholds
Da keine Ground-Truth existiert, werden Ereignisse über globale, datengetriebene Schwellenwerte erzeugt.

Prinzip:
- kritische Ereignisse (z. B. hoher Jerk, hohe Off-Road-Quote)
- akkumulative Scores für subtile Muster
- zeitliche Persistenz (Rolling-Max über 2 Sekunden)
- Prioritätsregel: **Aggressive > Inattentive > Attentive**

Formale Skizze:
- `S_aggressive = 0.3*I(accel_std>t1) + 0.3*I(steering_rate>t2) + 0.2*I(throttle_changes>t3) + 0.2*I(hr>t4)`
- `S_inattentive = 0.4*I(lateral_std>t5) + 0.3*I(ndrt_error>t6) + 0.3*I(1-control_smoothness>0.5)`
- Event, falls Score ≥ 0.5

### 4.3 Klassifikationsmodell (LightGBM)
Finale Konfiguration (Auszug):
- `n_estimators=200`
- `max_depth=6`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `min_child_samples=20`
- `reg_alpha=0.1`, `reg_lambda=1.0`
- `class_weight='balanced'`

Feature-Scaling erfolgt über `StandardScaler`.

### 4.4 Vergleichsmodelle (Random Forest, GMM)
**Random Forest (Baseline):**
- 200 Bäume, begrenzte Tiefe, balancierte Klassengewichte

**GMM (explorativ, unüberwacht):**
- 3 Komponenten
- Cluster-zu-Label-Mapping per Hungarian-Algorithmus
- primär zur Einordnung gegenüber überwachten Verfahren

### 4.5 Trainings- und Testdaten
Dokumentierter finaler Datensatz:
- Samples: **5,037,534**
- Features: **37** (+ Metadaten wie `session_id`, `time`)
- Session-basierter 80/20-Split:
  - Train: **4,010,219**
  - Test: **1,027,315**

### 4.6 Hyperparameter und Kostenfunktion
Für die Multiklassenklassifikation optimiert LightGBM eine probabilistische Loss-Funktion (multiclass log loss). Die gewählten Parameter begrenzen Overfitting und stabilisieren den Klassenvergleich bei Ungleichverteilung.

---

## 5. Evaluation

### 5.1 Accuracy
Repräsentative Ergebnisse (Session-Split, dokumentierter Lauf):
- Train Accuracy: **0.9246**
- Test Accuracy: **0.9197**
- Train-Test-Gap: **+0.0049**

Der geringe Gap spricht für gute Generalisierung.

### 5.2 Precision, Recall, F1-Score
Test-Report:

| Klasse | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Attentive | 0.93 | 0.95 | 0.94 | 524,109 |
| Inattentive | 0.91 | 0.89 | 0.90 | 246,694 |
| Aggressive | 0.91 | 0.89 | 0.90 | 256,512 |
| Weighted Avg | 0.92 | 0.92 | 0.92 | 1,027,315 |

### 5.3 Konfusionsmatrix und Cohen’s Kappa
Konfusionsmatrix (Test):

| True \ Pred | Attentive | Inattentive | Aggressive |
|---|---:|---:|---:|
| Attentive | 497,937 | 10,009 | 16,163 |
| Inattentive | 22,882 | 218,734 | 5,078 |
| Aggressive | 15,490 | 12,865 | 228,157 |

Zusätzliche Kennzahl:
- **Cohen’s Kappa: 0.8698**

Interpretation:
- Attentive wird sehr stabil erkannt.
- Die schwierigste Trennung liegt zwischen Inattentive und den Nachbarklassen.

### 5.4 Einordnung von MSE/RMSE im Klassifikationskontext
MSE/RMSE sind primär Regressionsmetriken und wurden nicht als Hauptkriterien verwendet. Für dieses Projekt sind Accuracy, F1 und Cohen’s Kappa aussagekräftiger, da ein diskretes Multiklassenproblem vorliegt.

### 5.5 Analyse der Modellgewichte (Feature Importance)
Die Importance-Analyse (LightGBM) bestätigt den multimodalen Ansatz:
- Top-Features liegen bei Blickverhalten (`gaze_off_road_ratio`) sowie Fahrdynamik (`jerk_long_max`, `steering_rate`, `lateral_pos_std`).
- Physiologische Features liefern zusätzlichen, aber kleineren Beitrag.

---

## 6. Modelloptimierung und Validierung

### 6.1 Session-übergreifende Validierung
Neben dem festen Split wurde Leave-One-Session-Out Cross-Validation implementiert. Dadurch wird verhindert, dass Daten derselben Session gleichzeitig in Train und Test landen (Leakage-Risiko). Das Verfahren prüft die Übertragbarkeit auf unbekannte Sessions.

### 6.2 Laufzeit- und GPU-Analyse
Dokumentierte Benchmarks:

| Datensatzgröße | CPU (s) | GPU (s) | Speedup |
|---|---:|---:|---:|
| 100,000 | 6.05 | 6.05 | 1.00x |
| 1,000,000 | 16.56 | 16.40 | 1.01x |
| 5,000,000 | 95.09 | 80.41 | 1.18x |

Fazit: Der GPU-Vorteil wird bei sehr großen Datenmengen relevant.

### 6.3 Vergleich mit anderen Modellen
Aus den final dokumentierten Vergleichen:

| Metrik | LightGBM | Random Forest |
|---|---:|---:|
| Train Accuracy | 92.46% | 90.12% |
| Test Accuracy | 91.97% | 88.53% |
| Train-Test-Gap | +0.49% | +1.59% |
| Cohen’s Kappa | 0.8698 | 0.8124 |
| Training Time | 74.86 s | 174.77 s |

Damit war LightGBM im Projekt die beste überwachte Methode.

### 6.4 Szenarien, Robustheit und Einsatz von generativer KI
- **CTAG/Videodaten**: Diese Validierungen waren im gegebenen Projektumfang nicht als vollwertige finale Messreihe enthalten.
- **Generative KI**: Wurde unterstützend zur Strukturierung von Experimentideen, Fehleranalyse und Dokumentationsentwürfen verwendet, jedoch nicht als Quelle für Modellmetriken.
- **Robustheit**: Hauptlimitation bleibt die Labelqualität ohne externe Ground-Truth.

---

## 7. Schlussfolgerung
Die Arbeit zeigt, dass eine belastbare Fahrerzustandsklassifikation aus unlabeled Simulator-Rohdaten möglich ist, wenn Labeling, Feature-Engineering und Evaluationsdesign konsistent aufeinander abgestimmt sind. Entscheidend war die Kombination aus:
- absolut-schwellenwertbasiertem Labeling mit zeitlicher Persistenz,
- multimodalem Feature-Set,
- session-basierter Validierung,
- leistungsfähigem Boosting-Modell (LightGBM).

Mit **91.97% Test-Accuracy** und **Kappa 0.8698** wurde eine hohe Modellgüte erreicht. Im direkten Vergleich war LightGBM sowohl genauer als auch effizienter als der Random-Forest-Baseline-Ansatz.

Für Folgearbeiten sind vor allem externe Validierung mit echten Fahrdaten, partiell manuelle Re-Annotation und zeitmodellbasierte Ansätze (z. B. LSTM/Transformer) relevant.

---

## 8. Literaturverzeichnis
1. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems (NeurIPS 30)*.
2. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
3. McLachlan, G., Peel, D. (2000). *Finite Mixture Models*. Wiley.
4. Dong, Y., Hu, Z., Uchimura, K., Murayama, N. (2011). Driver Inattention Monitoring System for Intelligent Vehicles: A Review. *IEEE Transactions on Intelligent Transportation Systems*, 12(2), 596–614.
5. Reimer, B. (2009). Impact of Cognitive Task Complexity on Drivers’ Visual Attention and Driving Performance. *Transportation Research Record*.
6. Van der Maaten, L., Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.

---

## 9. Anhang

### 9.1 Arbeitspakete
Umsetzung entlang typischer Projektphasen:
1. Pipeline-Aufbau und Datenverständnis  
2. Synchronisation und Bereinigung  
3. Feature Engineering  
4. Labeling-Strategie  
5. Modelltraining und Baseline-Vergleich  
6. Evaluation und Visualisierung  
7. Validierung und Laufzeitanalyse  
8. Dokumentation und Präsentation

### 9.2 Codeübersicht
Relevante Dateien:
- `LightGBMPipeline/main.py`
- `LightGBMPipeline/data_processing.py`
- `LightGBMPipeline/FeatureEngineer.py`
- `LightGBMPipeline/AbsoluteThresholdLabeler.py`
- `LightGBMPipeline/DriverStateClassifier.py`
- `LightGBMPipeline/RFComparison.py`
- `LightGBMPipeline/UnsupervisedComparison.py`
- `LightGBMPipeline/visualize_data.py`

### 9.3 Datenspezifikation
Siehe `VTI_t3.2_columns.md` mit Spaltenübersicht für Ego-, Physiologie-, NDRT-, Umfelddaten und Metadaten.

### 9.4 Weitere Diagramme
Im Ordner `LightGBMPipeline/plots` befinden sich die im Projekt erzeugten Abbildungen zur Verteilung, Korrelation, Modellleistung und Clusterdarstellung.
