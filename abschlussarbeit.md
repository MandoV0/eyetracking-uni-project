Leibniz Universität Hannover  
Fakultät für Elektrotechnik und Informatik  
Institut für ...  

  

**Analyse und Klassifikation von Fahrerzuständen aus Eye-Tracking- und Fahrsimulator-Daten mittels Machine-Learning-Verfahren**  

  

Informatik-Projektarbeit  
eingereicht von:  
_Vorname Nachname_  
Matrikelnummer: _XXXXXXX_  

Betreuer:  
Prof. Dr. rer. nat. Jamal Raiyn  

Hannover, Februar 2025  

---

## Inhaltsverzeichnis

1. **Einführung**  
   1.1 State of the Art  
   1.2 Problemstellung  
   1.3 Motivation  
   1.4 Ziele  
2. **Daten und Methodik**  
   2.1 Datenerfassung  
   2.2 Parameter-Spezifikation  
   2.3 Reinigung fehlerhafter Daten  
   2.4 Interpolation fehlender Werte  
   2.5 Synchronisation der Sensorströme  
   2.6 Datenqualitätsanalyse  
3. **Feature Engineering & Explorative Analyse**  
   3.1 Statistische Features  
   3.2 Dynamische Features  
   3.3 Klassifikation von Fahrertypen  
   3.4 Kausalitäts- und Korrelationsanalyse  
   3.5 Visualisierung der Merkmalsverteilungen  
4. **Modellierung**  
   4.1 Labelgenerierung durch absolute Schwellenwerte  
   4.2 Klassifikationsmodelle (LightGBM & Random Forest)  
   4.3 Unüberwachtes Clustering mit GMM  
   4.4 Trainings- und Testdaten  
   4.5 Hyperparameter & Verlustfunktion  
5. **Evaluation**  
   5.1 Accuracy  
   5.2 Precision, Recall, F1-Score  
   5.3 Konfusionsmatrix & Cohen’s Kappa  
   5.4 Evaluationsmetriken für unüberwachtes Lernen  
6. **Modelloptimierung & Validierung**  
   6.1 Leave-One-Session-Out-Cross-Validation  
   6.2 Vergleich mit Basismodellen  
   6.3 GPU-Beschleunigung und Laufzeitanalyse  
7. **Schlussfolgerung und Ausblick**  
8. **Literaturverzeichnis**  
9. **Anhang**  

---

## 1. Einführung

### 1.1 State of the Art

Mit zunehmender Automatisierung im Straßenverkehr gewinnt die zuverlässige Erkennung des Fahrerzustands stark an Bedeutung. Fahrerüberwachungssysteme (Driver Monitoring Systems, DMS) werden heute vor allem zur Erkennung von Müdigkeit, Ablenkung oder aggressivem Fahrverhalten eingesetzt. Klassische Systeme basieren häufig auf einzelnen Signalen wie Lenkverhalten oder Fahrzeugdynamik, während moderne Ansätze multimodale Datenquellen wie Eye-Tracking, physiologische Signale und Umfelddaten kombinieren.  

Im Bereich der Mustererkennung kommen überwiegend Machine-Learning- und Deep-Learning-Verfahren zum Einsatz. Convolutional Neural Networks (CNNs) und Recurrent Neural Networks (RNNs) werden beispielsweise auf Kameradaten eingesetzt, um Blickrichtung, Lidöffnungsgrad oder Gesichtsausdruck zu analysieren. Gradient-Boosting-Methoden wie LightGBM haben sich indes als sehr effiziente und robuste Modelle für tabellarische Sensordaten etabliert und werden insbesondere in Szenarien mit vielen, teils korrelierten Features eingesetzt.  

Gleichzeitig steht die Forschung vor der Herausforderung, dass große, fein annotierte Datensätze teuer sind. In Fahrsimulatorstudien oder Eye-Tracking-Experimenten liegen häufig nur Rohdaten ohne explizite Labels für Zustände wie „Attentive“, „Inattentive“ oder „Aggressive“ vor. Ein zentraler Forschungsansatz ist daher die halb- oder schwach-supervisierte Labelgenerierung, etwa über heuristische Regeln oder physikalisch interpretierbare Schwellenwerte.  

### 1.2 Problemstellung

Im Rahmen des vorliegenden Informatikprojekts soll der Fahrerzustand in drei Klassen eingeteilt werden:

- **Attentive**: Fahrer ist aufmerksam, folgt der Spur stabil, reagiert angemessen auf Verkehrssituationen.  
- **Inattentive**: Fahrer ist abgelenkt oder schläfrig, z. B. durch Off-Road-Blicke, instabile Spurhaltung oder fehlerhafte Non-Driving-Related Tasks (NDRT).  
- **Aggressive**: Fahrer zeigt risikoreiches Fahrverhalten mit starken Beschleunigungen, heftigen Lenkbewegungen oder sehr geringen Abständen.  

Die wesentlichen Herausforderungen sind:

- Es existieren **keine manuellen Ground-Truth-Labels**, die direkt für das Training eines Modells genutzt werden können.  
- Die Daten stammen aus verschiedenen Sensorströmen (Fahrsimulator, Eye-Tracking, Physiologie, Umfelddaten), die zunächst **synchronisiert, interpoliert und bereinigt** werden müssen.  
- Das Modell soll trotz großer Datenmengen (über 5 Mio. Zeitschritte) **robust generalisieren** und nicht nur spezifische Fahrsessions oder Fahrer „auswendig lernen“.  

### 1.3 Motivation

Ein zuverlässiges Modell zur Erkennung von Fahrerzuständen ist sowohl wissenschaftlich als auch praktisch relevant.  

Wissenschaftlich ermöglicht eine systematische Analyse, welche Merkmale (z. B. Eye-Tracking-Kennwerte, Spurhaltung, Herzrate) besonders stark mit Aufmerksamkeits- oder Aggressivitätszuständen korrelieren. Dies trägt zum Verständnis bei, welche sensorischen Größen für zukünftige DMS-Systeme priorisiert werden sollten.  

Praktisch könnten die hier entwickelten Methoden als Grundlage für Fahrerassistenzsysteme dienen, die in Echtzeit auf kritische Zustände reagieren – etwa durch Warnungen bei Inattention oder adaptives Anpassen von Fahrerassistenzfunktionen bei aggressivem Fahrstil. Die Nutzung eines effizienten, tabellenbasierten Machine-Learning-Ansatzes wie LightGBM ist zudem attraktiv, weil er gut in bestehende eingebettete Systeme integrierbar ist.  

### 1.4 Ziele

Die Ziele dieser Arbeit lassen sich wie folgt zusammenfassen:

- **Z1 – Datenpipeline**: Aufbau einer robusten Pipeline zur **Synchronisation, Aufbereitung und Feature-Generierung** aus Fahrsimulator-, Eye-Tracking-, Physiologie- und Umfelddaten.  
- **Z2 – Labelgenerierung**: Entwicklung eines Labeling-Verfahrens auf Basis **absoluter, physikalisch interpretierbarer Schwellenwerte**, um aus unlabeled Daten die drei Fahrerzustände abzuleiten.  
- **Z3 – Modellierung**: Training eines **LightGBM-Klassifikators** als Hauptmodell zur Vorhersage von „Attentive“, „Inattentive“ und „Aggressive“ sowie Implementierung eines **Random-Forest-Baseline-Modells**.  
- **Z4 – Unüberwachter Vergleich**: Einsatz eines **Gaussian-Mixture-Modells (GMM)** als unüberwachtes Verfahren, um zu untersuchen, inwiefern sich die drei Fahrzustände ohne Labels als Clusterstruktur wiederfinden.  
- **Z5 – Evaluation & Validierung**: Systematische Evaluation der Modelle mittels Accuracy, Precision, Recall, F1-Score und Cohen’s Kappa sowie Überprüfung der Generalisierungsfähigkeit über unterschiedliche Fahrsessions hinweg.  

---

## 2. Daten und Methodik

### 2.1 Datenerfassung

Die Daten stammen aus einer Fahrsimulatorstudie (Verzeichnis `data/VTI/T3.2`), in der Probanden verschiedene Fahrszenarien mit Eye-Tracking und zusätzlicher Physiologie erfasst haben. Pro Teilnehmer werden mehrere CSV-Dateien gespeichert:

- **Ego-Fahrzeugdaten** (`Ego vehicle data`): Fahrzeugdynamik (Position, Geschwindigkeit, Beschleunigung, Lenk- und Bremssignale).  
- **Physiologische Daten** (`Physiology`): EKG-Signal und daraus abgeleitete Herzrate und RR-Intervalle.  
- **Umfelddaten** (`Surrounding vehicle data`): Position, Geschwindigkeit und Typ umliegender Verkehrsteilnehmer.  

Die Funktion `find_pairs` im Modul `data_processing.py` verknüpft diese Dateien anhand einer gemeinsamen ID zu **Sessions** und liefert Tripel `(df_ego, df_phys, df_surround)`. Jede Session beschreibt einen zusammenhängenden Fahrverlauf eines Teilnehmers.  

### 2.2 Parameter-Spezifikation

Die Rohdaten umfassen unter anderem folgende Größen:

- **Fahrzeugdynamik**:  
  - `oveBodyAccelerationLongitudinalX`, `oveBodyAccelerationLateralY` (Längs- und Querbeschleunigung)  
  - `oveBodyJerkLongitudinalX`, `oveBodyJerkLateralY` (Längs- und Quer-Jerk)  
  - `oveBodyVelocityX`, `oveBodyVelocityY` (Geschwindigkeitskomponenten)  
  - `steeringWheelAngle` (Lenkradwinkel), `throttle` (Gaspedalstellung), `brakePedalActive` (Bremsbetätigung)  
- **Spur- und Positionsdaten**:  
  - `ovePositionLateralR` (laterale Position relativ zur Fahrspur)  
  - `oveInertialPositionX/Y/Z` (absolute Positionen, u. a. zur Distanzberechnung zu anderen Fahrzeugen)  
- **Eye-Tracking**:  
  - `openxrGazeHeading`, `openxrGazePitch` (Blickrichtung in horizontaler/vertikaler Achse)  
  - `varjoEyelidOpening` (Lidöffnungsgrad)  
- **NDRT-Leistung** (Non-Driving-Related Tasks):  
  - `arrowsCorrectCount`, `arrowsWrongCount`, `arrowsTimeoutCount`  
- **Physiologie**:  
  - Rohsignal `ecg`, abgeleitet `heartRate` und `rrInterval`  
- **Umfelddaten**:  
  - Position und Geschwindigkeit umliegender Akteure, aus denen `surround_actor_count`, `surround_min_dist` und `surround_avg_speed` abgeleitet werden.  

### 2.3 Reinigung fehlerhafter Daten

Die Rohdaten enthalten Messrauschen, gelegentlich fehlende Werte sowie potenziell fehlerhafte Einträge (z. B. `-1` als Platzhalter). Im Modul `data_processing.py` werden deshalb folgende Schritte durchgeführt:

- **Parsing und Typkonvertierung**: Einlesen der CSV-Dateien mit `pandas`, explizites Casting relevanter Spalten in Fließkommazahlen.  
- **Bereinigung von Platzhalterwerten**: Beispielsweise wird `aheadTHW = -1` als fehlender Wert interpretiert und in der Feature-Berechnung explizit ersetzt.  
- **Entfernung physikalisch unrealistischer Werte**: In der Verarbeitung der Umfelddaten werden sehr kleine Distanzen (`dist <= 0.1 m`) verworfen, um Artefakte zu vermeiden.  
- **Einfache Peak-Detektion im EKG**: In `process_physiology_data` werden Herzschläge über eine Schwellwertlogik auf dem EKG-Signal detektiert, daraus `rrInterval` und `heartRate` berechnet und fehlende Werte anschließend per Vorwärts-/Rückwärtsfüllung ersetzt.  

### 2.4 Interpolation fehlender Werte

Die drei Sensorströme (Ego, Physiologie, Umfelddaten) besitzen unterschiedliche Abtastraten und Zeitstempel. Um ein einheitliches Zeitschema zu erhalten, wird in `sync_all` eine **Interpolation** auf die Zeitachse des Ego-Fahrzeugs vorgenommen.  

Es werden drei Modi unterstützt:

- **MERGE (0)**: `merge_asof` mit nächstgelegenem Zeitstempel innerhalb einer Toleranz.  
- **LINEAR (1)**: lineare Interpolation mittels `scipy.interpolate.interp1d`.  
- **SPLINE (2)**: kubische Spline-Interpolation.  

In der Pipeline wird standardmäßig der **lineare Modus** verwendet, da er einen guten Kompromiss zwischen Glättung und Rechenaufwand bietet. Fehlende Werte werden anschließend in den Feature-Tabellen per `bfill`/`ffill` sowie `fillna(0)` behandelt, um vollständige Vektoren als Input für das Modell zu erhalten.  

### 2.5 Synchronisation der Sensorströme

Die Funktion `sync_all` übernimmt die Zusammenführung der Ego-, Physiologie- und Umfelddaten auf Basis des Zeitstempels `time`. Der Ablauf ist:

1. Sortierung der Ego-Daten nach `time` und Kopie in ein zentrales DataFrame `merged`.  
2. Verarbeitung der physiologischen Daten (`process_physiology_data`) und zeitliche Ausrichtung via `merge_asof` oder Interpolation.  
3. Verknüpfung der Umfelddaten mit der Ego-Position, Berechnung der euklidischen Distanz zu jedem Akteur und Aggregation pro Zeitstempel (`surround_actor_count`, `surround_min_dist`, `surround_avg_speed`).  
4. `merge_asof` der aggregierten Umfelddaten mit `merged`.  

Damit steht für jede Zeitstufe einer Session ein konsistenter Vektor aus Fahrverhalten, Eye-Tracking, Physiologie und Umfelddaten zur Verfügung.  

### 2.6 Datenqualitätsanalyse

Zur Beurteilung der Datenqualität werden mit `visualize_data.py` verschiedene Diagramme erzeugt:

- **Histogramme der Rohdaten** (Beschleunigung, Lenkung, Geschwindigkeit, Herzrate, Blickrichtung, Lidöffnungsgrad), um Ausreißer und Verteilungen zu prüfen.  
- **Zeitreihenplots** einzelner Sessions, z. B. des Verlaufs von Beschleunigung, Gaze-Off-Road-Ratio, Herzrate und zugehörigen Labels.  

Die Analysen zeigen, dass die Sensoren stabile Verteilungen besitzen und die abgeleiteten physiologischen Größen plausibel variieren. Gleichzeitig wird sichtbar, dass extreme Fahrmanöver und starke Blickabweichungen relativ selten sind, was für die spätere Klassifikation (Klassenungleichgewicht) relevant ist.  

---

## 3. Feature Engineering & Explorative Analyse

### 3.1 Statistische Features

Das Feature Engineering wird im Modul `FeatureEngineer.py` durchgeführt und basiert auf **gleitenden Fenstern** mit einer Standardgröße von 50 Zeitschritten. Für jedes Fenster werden statistische Kennwerte (Mittelwert, Standardabweichung, Maximum, Summe) berechnet. Beispiele:

- **Beschleunigung und Jerk**:  
  - `accel_long_mean`, `accel_long_std`, `accel_long_max`  
  - `accel_lat_std`  
  - `jerk_long_mean`, `jerk_long_max`, `jerk_lat_std`  
- **Geschwindigkeit und Gas/Bremse**:  
  - `speed`, `speed_std`, `throttle_mean`, `throttle_changes`  
- **Lenkverhalten**:  
  - `steering_angle_std`, `steering_rate`, `yaw_velocity_std`  
- **Spurhaltung**:  
  - `lateral_pos_std`, `lateral_pos_mean`  
- **Physiologie**:  
  - `hr_mean`, `hr_change`, `hrv_sdnn`, `hrv_rmssd`  

Durch die Fensterung werden kurzzeitige Fluktuationen geglättet und mittel- bis langfristige Muster im Fahrverhalten hervorgehoben, die für die Klassifikation der Fahrerzustände relevant sind.  

### 3.2 Dynamische Features

Neben rein statistischen Kennwerten werden auch dynamische und temporale Aspekte modelliert:

- **Jerk-basierte Aggressivität**: Hohe Werte von `jerk_long_max` deuten auf abruptes Bremsen oder Beschleunigen hin.  
- **Lenkdynamik**: `steering_rate` erfasst die Änderungsgeschwindigkeit des Lenkradwinkels und damit ruckartige Lenkmanöver.  
- **Abstand zum Vordermann**: Über `aheadTHW` und `short_headway_ratio` wird das Verhältnis von Zeitabstand (Time Headway) im kritischen Bereich (z. B. < 2 s) bestimmt.  
- **Umfeldkomplexität**: `surround_actor_count` und `surround_min_dist` charakterisieren Verkehrsdichte und Nähe anderer Fahrzeuge.  
- **Kontrollglätte**: `control_smoothness` kombiniert Lenk- und Gasdynamik zu einem Maß für sanftes versus hektisches Fahren.  

### 3.3 Klassifikation von Fahrertypen (Feature-Ebene)

Bereits auf Feature-Ebene kann eine Zuordnung zu Fahrertypen vorgenommen werden:

- **Aggressive Fahrer** zeigen typischerweise hohe Werte bei `accel_long_std`, `jerk_long_max`, `steering_rate`, `throttle_changes` und häufig geringe Abstände (`short_headway_ratio` hoch).  
- **Inattentive Fahrer** weisen erhöhte `lateral_pos_std`, hohe `gaze_off_road_ratio`, viele NDRT-Fehler (`ndrt_error_rate`) und niedrigere `control_smoothness` auf.  
- **Attentive Fahrer** liegen in diesen Indikatoren meist im moderaten Bereich, mit stabiler Spurhaltung, geringer Blickabweichung und glatten Fahrzeugmanövern.  

Diese Intuition fließt unmittelbar in die spätere Labelgenerierung und dient auch im Rahmen der explorativen Analyse als Interpretationshilfe.  

### 3.4 Kausalitäts- und Korrelationsanalyse

Eine strenge kausale Analyse (z. B. Granger-Kausalität) wurde im Projektumfang nicht durchgeführt. Stattdessen wurde eine **Korrelationsanalyse** zentraler Merkmale genutzt, um Zusammenhänge zu identifizieren:

- Mit `plot_feature_correlation` wird eine Heatmap für ausgewählte Features (z. B. `jerk_long_max`, `steering_rate`, `gaze_off_road_ratio`, `ndrt_error_rate`, `hr_mean`, `control_smoothness`) erzeugt.  
- Es zeigt sich, dass aggressive Merkmale (Jerk, Lenkdynamik, Throttle-Änderungen) stark miteinander korrelieren, während Inattention-Indikatoren (Gaze-Off-Road, Spurabweichung, NDRT-Fehler) eine zweite, teilweise unabhängige Gruppe bilden.  

Diese Struktur unterstützt die Annahme, dass „Aggressiv“ und „Inattentive“ unterschiedliche Verhaltensmuster darstellen, die durch geeignete Feature-Kombinationen trennbar sind.  

### 3.5 Visualisierung der Merkmalsverteilungen

Zur Validierung der Feature-Definition und später der Labels werden mit `visualize_data.py` verschiedene Plots erzeugt:

- **Verteilungen pro Klasse**: `plot_feature_distributions` zeigt für ausgewählte Features die Histogramme getrennt nach Label (`Attentive`, `Inattentive`, `Aggressive`).  
- **Zeitreihenbeispiele**: `plot_time_series_example` visualisiert den zeitlichen Verlauf von Schlüsselmerkmalen zusammen mit den zugehörigen Labels für eine Session.  

Die Plots bestätigen, dass sich die Klassen auf Feature-Ebene sinnvoll unterscheiden, z. B. höhere `gaze_off_road_ratio` bei Inattention und stärkere Jerk-/Beschleunigungspeaks bei Aggression.  

---

## 4. Modellierung

### 4.1 Labelgenerierung durch absolute Schwellenwerte

Da keine manuell annotierten Labels vorhanden sind, wird im Modul `AbsoluteThresholdLabeler.py` ein Labeling-Ansatz implementiert, der auf **absoluten, physikalisch interpretierbaren Schwellenwerten** basiert. Ziel ist es, zirkuläre Logik zu vermeiden, bei der das Modell lediglich vom verwendeten Labeling-Verfahren „lernt“.  

Der Ansatz besteht aus zwei Komponenten:

- **Datengetriebene Schwellwertableitung**: In `_analyze_data_distribution` werden für ausgewählte Features hohe Perzentile bestimmt (z. B. 85.–90. Perzentil für `jerk_long_max`, `accel_long_std`, `steering_rate`, `throttle_changes`, `lateral_pos_std`, `gaze_off_road_ratio`, `ndrt_error_rate`, `eyelid_low_ratio`). Für die Herzrate wird ein z-Wert-basiertes Kriterium verwendet (`hr_mean > mean + 1.5·std`). Daraus entstehen globale Schwellen wie `jerk_high`, `steering_rate_high`, `gaze_off_road_high` etc.  
- **Ereignisbasierte Labelzuweisung mit Persistenz**:  
  - **Kritische Inattention**: Über-Threshold-Ereignisse bei `gaze_off_road_ratio` und `eyelid_low_ratio` setzen die Bool-Variable `inattentive_events`.  
  - **Kritische Aggression**: Sehr hohe Jerk-Werte (`jerk_long_max > jerk_high`) führen zu `aggressive_events`.  
  - **Akkumulative Scores**: Zusätzlich werden **Aggressions-** (`aggression_score`) und **Inattention-Scores** (`inattention_score`) gebildet, die gewichtete Beiträge verschiedener Indikatoren aufsummieren. Wird ein Schwellenwert (z. B. 0.5) überschritten, wird das jeweilige Ereignis-Flag gesetzt.  
  - **Zeitliche Glättung**: Ereignisse werden über ein Fenster von z. B. 2 s (bei 10 Hz → 20 Samples) per Rolling-Max geglättet, um kurzzeitige Peaks zu minimieren und stabile Zustände zu erzeugen.  

Die finale Labelzuordnung erfolgt mit Priorität:

- `Aggressive (2)`: wenn `aggressive_events` wahr ist.  
- `Inattentive (1)`: wenn `inattentive_events` wahr ist und nicht gleichzeitig `aggressive_events`.  
- `Attentive (0)`: in allen übrigen Fällen.  

Der Labeler gibt neben den diskreten Labels auch die Scores zurück, was eine spätere Analyse der Labelqualität ermöglicht.  

### 4.2 Klassifikationsmodelle (LightGBM & Random Forest)

Das zentrale Klassifikationsmodell ist ein **LightGBM-Gradient-Boosting-Modell**, implementiert in `DriverStateClassifier.py`:

- `LGBMClassifier` mit u. a.  
  - `n_estimators = 200`, `max_depth = 6`, `learning_rate = 0.05`  
  - `subsample = 0.8`, `colsample_bytree = 0.8`  
  - `min_child_samples = 20`, `reg_alpha = 0.1`, `reg_lambda = 1.0`  
  - `class_weight = 'balanced'` (zum Umgang mit Klassenungleichgewichten)  
  - `device = 'cpu'` oder `device = 'gpu'` (optional)  

Vor dem Training werden die ausgewählten Features (`feature_cols`) mit einem `StandardScaler` normiert.  

Als Baseline dient ein **Random Forest** (Modul `RFComparison.py`):

- `RandomForestClassifier` mit  
  - `n_estimators = 200`, `max_depth = 5`  
  - `min_samples_split = 30`, `min_samples_leaf = 25`  
  - `max_features = 'sqrt'`, `class_weight = 'balanced'`, `n_jobs = -1`  

Die Random-Forest-Architektur ist bewusst konservativ regularisiert, um Overfitting zu begrenzen und einen fairen Vergleich zu LightGBM zu ermöglichen.  

### 4.3 Unüberwachtes Clustering mit GMM

Zur Bewertung eines unüberwachten Ansatzes wird im Modul `UnsupervisedComparison.py` ein **Gaussian-Mixture-Modell (GMM)** verwendet:

- `GaussianMixture` mit `n_components = 3` (entsprechend den drei Fahrerzuständen), `covariance_type = 'full'`, `max_iter = 300`.  
- Training erfolgt auf skalierten Features (`StandardScaler` wie bei den überwachten Modellen).  

Da GMM keine Labels nutzt, wird zur **Zuordnung der Cluster zu Klassen** das `Hungarian`-Verfahren (`linear_sum_assignment`) eingesetzt. Die Cluster-Label-Zuordnung maximiert die Übereinstimmung zwischen Cluster-IDs und den aus dem Threshold-Labeler stammenden Pseudo-Labels, sodass im Nachhinein Accuracy, F1-Score, Adjusted Rand Index (ARI) und Normalized Mutual Information (NMI) berechnet werden können.  

### 4.4 Trainings- und Testdaten

Die Trainingsdaten werden über `build_processed_from_raw` bzw. `load_processed_csv` erzeugt und bestehen nach vollständiger Verarbeitung aus:

- **≈ 5 037 534 Samples** (Zeitschritte),  
- **37 Features** (ohne Metadaten wie `time`, `session_id`, `aggression_score`, `inattention_score`).  

Für die Evaluation werden zwei Strategien genutzt:

- **Session-basierter Train/Test-Split** (`session_train_test_split` in `main.py`):  
  - Aufteilung der eindeutigen `session_id`s in Trainings- und Test-Sessions, um Datenleakage zwischen Sessions zu vermeiden.  
  - Typischer Split: 80 % Training, 20 % Test.  
- **Leave-One-Session-Out-Cross-Validation** (`LeaveOneGroupOut` in `DriverStateClassifier.py`):  
  - In jeder Falte wird eine komplette Session als Testset verwendet, alle anderen Sessions bilden das Trainingsset.  
  - Dadurch wird geprüft, wie gut das Modell auf bisher unbekannte Fahrer/Sessions generalisiert.  

### 4.5 Hyperparameter & Verlustfunktion

LightGBM optimiert eine **multiklassige Log-Loss-Funktion** mit `num_class = 3`. Wichtige Aspekte:

- **Klassenbalancierung**: Durch `class_weight = 'balanced'` werden seltenere Zustände (z. B. stark aggressive oder stark inattentive Phasen) höher gewichtet.  
- **Regularisierung**: Tiefe (`max_depth`), Blattgröße (`min_child_samples`) und L1-/L2-Regularisierung (`reg_alpha`, `reg_lambda`) begrenzen Overfitting.  
- **Subsampling**: `subsample` und `colsample_bytree` reduzieren Korrelationen zwischen Bäumen und fördern Generalisierung.  

Für den Random Forest werden die Hyperparameter so gewählt, dass die Bäume flacher und die Blätter größer sind, um Overfitting zu vermeiden.  

Beim GMM sind die Hyperparameter hauptsächlich die Anzahl der Komponenten und die Art der Kovarianzmatrix. Mit `covariance_type = 'full'` werden die Korrelationen zwischen Features innerhalb eines Clusters explizit modelliert.  

---

## 5. Evaluation

### 5.1 Accuracy

Die zentrale Kennzahl für den Klassifikator ist die **Genauigkeit (Accuracy)**, definiert als Anteil korrekt klassifizierter Zeitschritte.  

Für das LightGBM-Modell ergibt sich in einem repräsentativen Run auf dem vollständigen Datensatz (`≈ 5 Mio. Samples`, 80/20-Session-Split) folgendes Bild (vgl. Kommentar in `main.py`):

- **Train Accuracy**: ca. 0,9246  
- **Test Accuracy**: ca. 0,9197  
- **Train–Test-Gap**: ca. +0,0049  

Der geringe Gap deutet darauf hin, dass das Modell kaum überfitten und gut auf bisher nicht gesehenen Sessions generalisieren kann.  

### 5.2 Precision, Recall, F1-Score

Zur detaillierteren Bewertung werden **Precision**, **Recall** und **F1-Score** pro Klasse berechnet. Für den oben beschriebenen Run (Testset mit ≈ 1 027 315 Samples) zeigt der Klassifikationsbericht:

- **Attentive**:  
  - Precision ≈ 0,93  
  - Recall ≈ 0,95  
  - F1-Score ≈ 0,94  
- **Inattentive**:  
  - Precision ≈ 0,91  
  - Recall ≈ 0,89  
  - F1-Score ≈ 0,90  
- **Aggressive**:  
  - Precision ≈ 0,91  
  - Recall ≈ 0,89  
  - F1-Score ≈ 0,90  

Die **gewichteten** und **makro-gemittelten** Kennzahlen liegen jeweils um **0,91–0,92**, was eine insgesamt hohe Klassifikationsgüte über alle drei Zustände hinweg bestätigt. Besonders wichtig ist, dass auch die sicherheitskritischen Klassen „Inattentive“ und „Aggressive“ solide F1-Werte erreichen.  

### 5.3 Konfusionsmatrix & Cohen’s Kappa

Die **Konfusionsmatrix** des LightGBM-Modells für das Testset zeigt u. a.:

- Viele korrekte Zuordnungen entlang der Diagonale (z. B. ≈ 497 937 korrekt erkannte „Attentive“-Samples).  
- Verwechslungsfehler treten vor allem zwischen **„Inattentive“** und **„Aggressive“** auf, was plausibel ist, da beide Zustände von „normalem“ Fahren abweichen und sich in Grenzsituationen überlappen können.  

Zur robusteren Einschätzung wird **Cohen’s Kappa** berechnet. Für das Testset ergibt sich:

- **Cohen’s Kappa**: ≈ 0,87  

Ein Kappa-Wert in dieser Größenordnung gilt in der Literatur als **„sehr gute Übereinstimmung“** über dem Zufallsniveau und bestätigt, dass das Modell nicht nur triviale oder durch Klassenverteilung getriebene Vorhersagen trifft.  

### 5.4 Evaluationsmetriken für unüberwachtes Lernen

Für das GMM-Clustering werden neben Accuracy (nach Mapping per Hungarian-Algorithmus) auch unsupervisierte Kennzahlen betrachtet:

- **Adjusted Rand Index (ARI)**: misst die Übereinstimmung zwischen Clusterzuordnung und (Pseudo-)Labels, korrigiert um Zufall.  
- **Normalized Mutual Information (NMI)**: bewertet den Informationsgehalt der Clusterstruktur relativ zur Labelverteilung.  

Insgesamt zeigt sich, dass das GMM-Modell zwar eine grobe Trennung zwischen normalem und auffälligem Fahrverhalten realisieren kann, jedoch **deutlich hinter dem überwachten LightGBM-Modell** zurückbleibt – insbesondere bei der klaren Differenzierung von „Inattentive“ und „Aggressive“. Damit bestätigt sich, dass die Kombination aus durchdachtem Labeling und überwachten Modellen dem reinen Clustering überlegen ist.  

---

## 6. Modelloptimierung & Validierung

### 6.1 Leave-One-Session-Out-Cross-Validation

Zur Validierung der Generalisierungsfähigkeit wird in `DriverStateClassifier.train_with_cross_validation` ein **Leave-One-Session-Out-Ansatz** eingesetzt:

- Jede Fahrsession (Identifiers in `session_id`) wird einmal vollständig als Testset gehalten.  
- Alle übrigen Sessions bilden das Trainingsset.  
- Pro Falte werden Train- und Test-Accuracy, die Verteilung der Labels im Testset sowie der Train–Test-Gap ausgegeben.  

Über alle Falten hinweg liegen die Test-Accuracies eng beieinander, und der mittlere Train–Test-Gap bleibt klein. Dies deutet darauf hin, dass das Modell **nicht auf einzelne Fahrer überanpasst**, sondern tatsächlich allgemeine Muster im Fahr- und Blickverhalten lernt.  

### 6.2 Vergleich mit Basismodellen

Der in `RFComparison.py` implementierte Random Forest dient als Basismodell, um den Mehrwert von LightGBM zu quantifizieren.  

Die Experimente zeigen qualitativ:

- Der Random Forest erreicht eine gute, aber **etwas niedrigere Testgenauigkeit** als LightGBM.  
- Durch die begrenzte Tiefe und großen Blattgrößen ist der Random Forest robuster gegen Overfitting, lernt jedoch feinere Nichtlinearitäten in der Feature-Interaktion weniger gut als LightGBM.  

Das **LightGBM-Modell** stellt damit den besten Kompromiss aus **Genauigkeit, Robustheit und Effizienz** dar und wird als bevorzugtes Modell für die Fahrerzustandsklassifikation identifiziert.  

### 6.3 GPU-Beschleunigung und Laufzeitanalyse

Ein praktischer Aspekt für reale Anwendungen ist die **Trainings- und Inferenzzeit**. In `main.py` sind Benchmarks dokumentiert, die CPU- und GPU-Ausführung vergleichen (Ryzen 5 5600X vs. AMD RX 6700). Für das Training des LightGBM-Modells ergeben sich ungefähr:

- **100 000 Zeilen**: GPU ≈ 6,05 s (ähnlich CPU).  
- **1 000 000 Zeilen**: CPU ≈ 16,56 s, GPU ≈ 16,40 s.  
- **5 000 000 Zeilen**: CPU ≈ 95,09 s, GPU ≈ 80,41 s → ca. **15 % schneller**.  

Damit zeigt sich, dass GPU-Unterstützung insbesondere bei sehr großen Datensätzen einen signifikanten Vorteil bietet, während bei kleineren Datenmengen die Unterschiede vernachlässigbar sind. Für iteratives Prototyping reicht daher meist die CPU aus, für großskalige Trainingsläufe kann die GPU den Durchsatz erhöhen.  

---

## 7. Schlussfolgerung und Ausblick

In dieser Arbeit wurde eine vollständige Pipeline zur **Erkennung von Fahrerzuständen** aus Fahrsimulator-, Eye-Tracking-, Physiologie- und Umfelddaten entwickelt. Zentraler Bestandteil ist ein datengetriebenes Labeling auf Basis **absoluter, physikalisch interpretierbarer Schwellenwerte**, das aus unlabeled Rohdaten Pseudo-Labels für die Zustände „Attentive“, „Inattentive“ und „Aggressive“ erzeugt.  

Das darauf trainierte **LightGBM-Modell** erreicht eine **Testgenauigkeit von rund 92 %** bei gleichzeitig hohem Cohen’s Kappa und ausgewogenen F1-Scores über alle drei Klassen. Im Vergleich dazu liefert der Random Forest solide, aber leicht schlechtere Ergebnisse, während das unüberwachte GMM-Clustering ohne Labelinformation erwartungsgemäß deutlich hinter den überwachten Verfahren zurückbleibt.  

Für zukünftige Arbeiten bieten sich mehrere Erweiterungen an:

- Integration von **zeitlichen Modellen** (z. B. LSTMs oder Temporal Convolutional Networks), um Sequenzinformationen expliziter zu nutzen.  
- Verfeinerung des Labelers, etwa durch Kombination aus expertendefinierten Regeln und aktiven Lernstrategien mit partieller menschlicher Annotation.  
- Übertragung der Ansätze auf **realweltliche Fahrdaten** und Untersuchung der Robustheit gegenüber Sensorrauschen, Okklusionen und größeren Fahrerpopulationen.  

Insgesamt zeigt die Arbeit, dass bereits mit einem sorgfältig konstruierten Feature-Set und einem effizienten Gradient-Boosting-Modell eine **praxisnahe, robuste und interpretierbare Fahrerzustandsklassifikation** erreicht werden kann.  

---

## 8. Literaturverzeichnis

*(Die folgenden Einträge sind beispielhaft und sollten für die finale Abgabe ggf. noch an die tatsächlich verwendete Literatur angepasst werden.)*  

- Ke, G. et al. (2017): **LightGBM: A Highly Efficient Gradient Boosting Decision Tree**. In: Advances in Neural Information Processing Systems (NeurIPS).  
- Bishop, C. M. (2006): **Pattern Recognition and Machine Learning**. Springer.  
- Reimer, B. (2009): **Impact of Cognitive Task Complexity on Drivers’ Visual Attention**. Transportation Research Record.  
- Dong, Y., Hu, Z., Uchimura, K., Murayama, N. (2011): **Driver Inattention Monitoring System for Intelligent Vehicles: A Review**. IEEE Transactions on Intelligent Transportation Systems.  
- Sun, J., Yu, X., He, W. (2020): **Driver Distraction Detection Based on Deep Learning and Eye-Tracking**. IEEE Access.  
- McLachlan, G., Peel, D. (2000): **Finite Mixture Models**. Wiley.  

---

## 9. Anhang

- **A.1 Arbeitspakete und Projektplanung**  
  - Detaillierte Sprint- und Arbeitspaketplanung gemäß Vorgabe (siehe separate Planungssektion in `abschluss.md`).  
- **A.2 Code**  
  - Zentrale Module der Implementierung, insbesondere:  
    - `data_processing.py` (Laden, Synchronisation und Vorverarbeitung der Rohdaten)  
    - `FeatureEngineer.py` (Feature Engineering)  
    - `AbsoluteThresholdLabeler.py` (Labelgenerierung)  
    - `DriverStateClassifier.py` (LightGBM-Modell)  
    - `RFComparison.py`, `UnsupervisedComparison.py` (Vergleichsmodelle)  
    - `main.py`, `visualize_data.py` (Pipeline-Steuerung, Visualisierung)  
- **A.3 Datenspezifikation**  
  - Beschreibung der im Projekt verwendeten Rohdatenfelder und ihrer Einheiten (Fahrdynamik, Eye-Tracking, Physiologie, Umfelddaten).  
- **A.4 Weitere Diagramme**  
  - Zusätzliche Visualisierungen (Heatmaps, Zeitreihenplots, Feature-Distributionsplots), die im Projekt generiert, aber nicht im Haupttext abgedruckt wurden.  

