# Projektdokumentation: Driver State Classification (LightGBM Pipeline)

Autor: Burak Güldüz  
Projekt: Eye-Tracking Informatikprojekt (Fahrerzustandsklassifikation)  
Stand: März 2026

---

## 1. Ziel des Systems
Diese Pipeline klassifiziert den Fahrerzustand in drei Klassen:
- `0 = Attentive`
- `1 = Inattentive`
- `2 = Aggressive`

Die Besonderheit des Projekts: Es gibt keine manuellen Ground-Truth-Labels. Labels werden daher algorithmisch über eine regelbasierte Heuristik mit globalen Schwellenwerten erzeugt und danach für überwachte Modelle genutzt.

---

## 2. End-to-End Architektur
Die Verarbeitung läuft in folgenden Schritten:
1. Rohdaten laden (Ego, Physiologie, Surrounding Vehicles)  
2. Zeitlich synchronisieren (MERGE / LINEAR / SPLINE)  
3. Features berechnen (Rolling-Window, multimodal)  
4. Pseudo-Labels generieren (`AbsoluteThresholdLabeler`)  
5. Session-basiert train/test split oder LOSO-CV  
6. Modelltraining (LightGBM, RF; optional GMM als unsupervised Vergleich)  
7. Evaluation und Plot-Generierung

---

## 3. Projektstruktur (relevant)
- `LightGBMPipeline/main.py`  
  Zentrale CLI, Training, Evaluation, Plot-Steuerung
- `LightGBMPipeline/data_processing.py`  
  Rohdatenimport, Synchronisation, Cache-Build
- `LightGBMPipeline/FeatureEngineer.py`  
  Berechnet 37 Features
- `LightGBMPipeline/AbsoluteThresholdLabeler.py`  
  Generiert Labels mit Schwellenwerten + Persistenz
- `LightGBMPipeline/DriverStateClassifier.py`  
  LightGBM-Training, CV, Metriken
- `LightGBMPipeline/RFComparison.py`  
  Random-Forest-Baseline
- `LightGBMPipeline/UnsupervisedComparison.py`  
  GMM-Clustervergleich
- `LightGBMPipeline/visualize_data.py`  
  Diagramme für EDA und Ergebnisplots

---

## 4. Datenbasis und Formate
Erwartete Rohdatenstruktur:
- `data/VTI/T3.2/Ego vehicle data/`
- `data/VTI/T3.2/Physiology/`
- `data/VTI/T3.2/Surrounding vehicle data/`

Datenspalten-Referenz:
- `VTI_t3.2_columns.md`

Wichtige Inputs:
- Fahrdynamik: Beschleunigung, Jerk, Lenkung, Headway
- Eye-Tracking: Gaze Heading/Pitch, Eyelid Opening
- Physiologie: ECG, abgeleitet Heart Rate / RR Interval
- Umfeld: Distanz und Dichte umliegender Fahrzeuge

---

## 5. Setup
### 5.1 Python-Umgebung
Empfohlen:
- Python 3.10+
- Abhängigkeiten aus `requirements.txt` oder `reqs.txt`

Beispiel:
```powershell
pip install -r requirements.txt
```

### 5.2 Arbeitsverzeichnis
Für die Ausführung der Pipeline:
```powershell
cd LightGBMPipeline
```

---

## 6. Pipeline ausführen

### 6.1 Mit vorhandenem Cache (schneller)
```powershell
python main.py --eval-mode split --model-type lgbm --processed-csv cache/features_labels_full.csv --plot
```

### 6.2 Rohdaten neu verarbeiten und Cache bauen
```powershell
python main.py --from-raw --rebuild-cache --max-files 50 --window-size 50 --interpolation 1 --eval-mode split --model-type lgbm --plot
```

Parameter:
- `--interpolation 0|1|2`  
  `0=MERGE`, `1=LINEAR`, `2=SPLINE`
- `--eval-mode cv|split`  
  `cv`: Leave-One-Session-Out, `split`: session-basierter Holdout
- `--model-type lgbm|rf|gmm`
- `--use-gpu` (optional für große Datensätze)

### 6.3 Random-Forest-Vergleich
```powershell
python main.py --eval-mode split --model-type rf --processed-csv cache/features_labels_full.csv --plot
```

### 6.4 GMM Vergleich (unsupervised)
```powershell
python main.py --eval-mode split --model-type gmm --processed-csv cache/features_labels_full.csv --visualize-clusters
```

---

## 7. Labeling-Strategie
Komponente: `AbsoluteThresholdLabeler`

Prinzip:
- Datengetriebene globale Schwellenwerte (Perzentile über Gesamtdaten)
- Kritische Events (z. B. starker Jerk, hohe Off-Road-Ratio)
- Akkumulative Scores für Aggression/Inattention
- Temporale Persistenz (`event_persistence_seconds`, Standard 2s)
- Priorität: `Aggressive > Inattentive > Attentive`

Ziel:
- Stabilere, nachvollziehbare Pseudo-Labels trotz fehlender Ground-Truth
- Weniger Label-Flackern durch zeitliche Glättung

---

## 8. Feature Engineering (Kurzüberblick)
Komponente: `FeatureEngineer`

Featuregruppen:
- Aggressive-Indikatoren (z. B. `jerk_long_max`, `steering_rate`, `throttle_changes`)
- Inattentive-Indikatoren (z. B. `gaze_off_road_ratio`, `lateral_pos_std`, `eyelid_low_ratio`)
- Kontext-/Physiologie-Features (z. B. `surround_min_dist`, `hr_mean`, `hrv_rmssd`)

Fensterung:
- Rolling-Window Standard `window_size=50`

---

## 9. Modelle und Hyperparameter
### 9.1 LightGBM (Hauptmodell)
In `DriverStateClassifier.py`:
- `n_estimators=200`
- `max_depth=6`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `min_child_samples=20`
- `reg_alpha=0.1`, `reg_lambda=1.0`
- `class_weight='balanced'`

### 9.2 Random Forest (Baseline)
In `RFComparison.py`:
- `n_estimators=200`
- `max_depth=5`
- `min_samples_split=30`
- `min_samples_leaf=25`
- `max_features='sqrt'`
- `class_weight='balanced'`

### 9.3 GMM (Unsupervised Vergleich)
In `UnsupervisedComparison.py`:
- `n_components=3`
- `covariance_type='full'`
- Mapping zu Labels per Hungarian Algorithmus

---

## 10. Ergebnisse (dokumentierter finaler Lauf)
Datensatz:
- `5,037,534` Samples
- `37` Features
- Session-basierter Split: 80/20

LightGBM:
- Train Accuracy: `0.9246`
- Test Accuracy: `0.9197`
- Cohen's Kappa: `0.8698`

Random Forest:
- Train Accuracy: `0.9012`
- Test Accuracy: `0.8853`
- Cohen's Kappa: `0.8124`

Interpretation:
- LightGBM ist im Projektsetting konsistent besser als RF
- Train-Test-Gap bleibt klein, Hinweis auf gute Generalisierung

---

## 11. Ausgaben und Artefakte
Typische Artefakte:
- Modell: `LightGBMPipeline/driver_state_lgbm.pkl`
- Cache: `LightGBMPipeline/cache/features_labels_full.csv`
- Plots: `LightGBMPipeline/plots/*.png`
  - `label_distribution.png`
  - `confusion_matrix.png`
  - `feature_importance.png`
  - `feature_correlation.png`
  - `gmm_clusters_visualization.png`

---

## 12. Reproduzierbarkeit
Empfehlungen:
- Fixe Seeds (`--random-state 42`)
- Gleiche Split-Strategie (session-basiert)
- Exakte Versionierung von Python + Libraries dokumentieren
- Ergebnisse immer mit Command-Line und Datensatzgröße protokollieren

---

## 13. Bekannte Limitationen
- Keine manuell verifizierten Ground-Truth-Labels
- Simulatorverhalten ist nicht vollständig identisch zu Realverkehr
- Labelqualität hängt von Threshold-Definition und Sensorqualität ab
- Domänentransfer auf reale Fahrzeugdaten ist noch offen

---

## 14. Erweiterungen (Roadmap)
1. Teilweise manuelle Re-Annotation für Validierungs-Subset  
2. Zeitmodelle (LSTM/TCN/Transformer) für sequenzielle Abhängigkeiten  
3. Robustheitstests mit neuen Fahrern/Szenarien  
4. SHAP/XAI für tiefergehende Modellinterpretation  
5. Echtzeit-Deployment (Streaming-Inferenz)

---

## 15. Kurzer Betriebsleitfaden
Für schnelle Demo:
1. `cd LightGBMPipeline`
2. `python main.py --eval-mode split --model-type lgbm --processed-csv cache/features_labels_full.csv --plot`
3. Plots in `LightGBMPipeline/plots` prüfen
4. Bei Bedarf `--model-type rf` und `--model-type gmm` für Vergleich ausführen

