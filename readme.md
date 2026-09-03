# XAI (Explainable AI) for fail2drive

Dieses Modul erweitert das fail2drive Framework um Explainability-Funktionen basierend auf [Captum](https://captum.ai/). Es ermöglicht zu verstehen, **warum** das Modell bestimmte Fahrentscheidungen trifft - welche Bildbereiche, LiDAR-Punkte oder Eingaben eine Vorhersage beeinflussen.

## Installation

```bash
pip install captum>=0.7.0
```

Oder via requirements.txt (bereits enthalten):
```bash
pip install -r team_code/requirements.txt
```

---

## Überblick: Zwei Betriebsmodi

### Modus 1: Tensor-Speicherung bei Evaluation (empfohlen)

Während der normalen CARLA-Evaluation werden Modell-Inputs als `.pt` Dateien gespeichert. Die eigentliche XAI-Analyse erfolgt danach offline - beliebig oft wiederholbar mit verschiedenen Methoden.

**Vorteil**: Minimaler Overhead bei der Evaluation, volle Flexibilität bei der Analyse.

### Modus 2: Live XAI während der Evaluation

XAI-Heatmaps werden direkt während der Evaluation berechnet und gespeichert.

**Vorteil**: Sofortige Ergebnisse. **Nachteil**: Deutlich längere Evaluationszeit.

---

## Schritt-für-Schritt Anleitung

### 1. Evaluation mit Tensor-Speicherung ausführen

```bash
DEBUG_CHALLENGE=1 SAVE_PATH=./eval_output XAI_ENABLED=1 \
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --routes ./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml \
  --agent ./team_code/sensor_agent.py \
  --agent-config ./checkpoints/tfpp
```

Gespeicherte Tensoren finden sich unter:
```
./eval_output/<route_name>/xai_tensors/
  000000.pt    # Step 0
  000100.pt    # Step 100
  000200.pt    # Step 200
  ...
```

Jede `.pt` Datei enthält alle Modell-Inputs eines Zeitschritts:
- `rgb` - Kamerabild (3, 384, 1024), ImageNet-normalisiert
- `lidar_bev` - LiDAR Bird's Eye View (C, 256, 256)
- `target_point` - Navigationsziel in Ego-Koordinaten (2,)
- `ego_vel` - Fahrzeuggeschwindigkeit (1,)
- `command` - Navigationskommando, one-hot (6,)
- `step` - Zeitschritt-Nummer

### 2. Offline-Analyse durchführen

```bash
cd team_code

# Schnelltest mit Saliency (wenige Sekunden pro Sample):
python -m xai.analysis \
    --checkpoint ../checkpoints/tfpp \
    --tensor_dir ../eval_output/<route>/xai_tensors \
    --output_dir ../xai_results \
    --methods saliency \
    --output_heads target_speed \
    --num_samples 5

# Vollständige Analyse:
python -m xai.analysis \
    --checkpoint ../checkpoints/tfpp \
    --tensor_dir ../eval_output/<route>/xai_tensors \
    --output_dir ../xai_results \
    --methods saliency integrated_gradients grad_cam deeplift \
    --output_heads target_speed checkpoint waypoint \
    --attention

# Nur bestimmte Anzahl Samples:
python -m xai.analysis \
    --checkpoint ../checkpoints/tfpp \
    --tensor_dir ../eval_output/<route>/xai_tensors \
    --output_dir ../xai_results \
    --methods integrated_gradients \
    --output_heads target_speed checkpoint \
    --num_samples 20
```

### 3. Ergebnisse

```
xai_results/
  step_000000/
    saliency_target_speed_000000.png          # Saliency-Heatmap für Speed
    integrated_gradients_target_speed_000000.png
    grad_cam_checkpoint_000000.png
    attention_flow_000000.png                  # Cross-modal Attention
  step_000100/
    ...
  summary.json   # Aggregierte Statistiken (RGB vs LiDAR Importance)
```

---

## Environment-Variablen

| Variable | Werte | Beschreibung |
|----------|-------|--------------|
| `XAI_ENABLED` | `0` / `1` | Aktiviert Tensor-Speicherung bei Evaluation |
| `XAI_LIVE` | `0` / `1` | Zusätzlich Live-XAI-Berechnung während Evaluation |
| `XAI_METHOD` | `saliency`, `integrated_gradients`, `grad_cam`, `feature_ablation`, `deeplift` | Methode für Live-Modus |
| `XAI_OUTPUT_HEAD` | siehe unten | Output-Head für Live-Modus |
| `SAVE_PATH` | Pfad | Speicherort für alle Outputs |
| `DEBUG_CHALLENGE` | `0` / `1` | Aktiviert Debug-Visualisierungen |

---

## Verfügbare XAI-Methoden

| Methode | ID | Beschreibung | Geschwindigkeit |
|---------|-----|--------------|-----------------|
| Saliency | `saliency` | Gradient des Outputs bzgl. der Inputs. Zeigt welche Pixel bei kleiner änderung den grössten Einfluss haben. | Sehr schnell |
| Integrated Gradients | `integrated_gradients` | Akkumuliert Gradienten entlang eines Pfades von einem Baseline-Input zum tatsächlichen Input. Erfüllt Axiome der Attribution (Completeness, Sensitivity). | Langsam (n_steps=50) |
| GradCAM | `grad_cam` | Gewichtete Aktivierungskarte der letzten Convolutional Layer. Zeigt welche räumlichen Regionen im Feature-Space relevant sind. | Schnell |
| Feature Ablation | `feature_ablation` | Systematisches Entfernen von Input-Regionen und Messen des Output-Unterschieds. | Mittel |
| DeepLift | `deeplift` | Vergleicht Neuronen-Aktivierungen mit einer Referenz-Aktivierung (Baseline). Verwendet modifizierte Backpropagation-Regeln statt Standard-Gradienten. | Schnell |
| Attention | `--attention` Flag | Extrahiert Cross-Modal Attention-Weights aus den GPT Fusion-Blöcken (Image <-> LiDAR). | Schnell |

---

## Erklärbare Output-Heads

| Output Head | ID | Was wird erklärt |
|-------------|-----|-------------------|
| Target Speed | `target_speed` | Warum sagt das Modell eine bestimmte Geschwindigkeit vorher? |
| Checkpoints | `checkpoint` | Welche Inputs beeinflussen die vorhergesagten Routenpunkte? |
| Waypoints | `waypoint` | Was treibt die Waypoint-Vorhersage (Lenkverhalten)? |
| Bounding Boxes | `bbox` | Worauf basiert die Objekterkennung? |
| Semantic Seg. | `semantic` | Was beeinflusst die semantische Segmentierung? |

---

## Konfiguration (config.py)

In `team_code/config.py` (`GlobalConfig`) sind folgende Parameter definiert:

```python
self.xai_enabled = False          # Master-Switch (wird durch Env-Var überschrieben)
self.xai_methods = [...]          # Verfügbare Methoden
self.xai_output_heads = [...]     # Verfügbare Output-Heads
self.xai_save_freq = 10           # Speichere Tensoren alle N Steps
self.xai_n_steps = 50             # Integrated Gradients Schritte
self.xai_baseline = 'zeros'       # Baseline für IG (zeros = schwarzes Bild)
self.xai_target_layer = '...'     # Ziel-Layer für GradCAM
self.xai_skip_first_steps = 60    # Erste N Steps überspringen (Warmup)
self.xai_event_driven = True      # Event-basiertes Speichern aktivieren
self.xai_event_object_distance = 15.0  # Speichern wenn Objekt näher als N Meter
self.xai_event_brake_threshold = 1.0   # Speichern wenn pred. Speed < N m/s
```

### Smart Saving: Event-basiertes Speichern

Zusätzlich zum periodischen Speichern (alle N Steps) werden Tensoren automatisch
gespeichert wenn relevante Events auftreten:

- **Objekt in der Nähe**: Ein erkanntes Objekt (Fahrzeug, Fussgänger, etc.) ist
  näher als `xai_event_object_distance` Meter
- **Bremsvorhersage**: Das Modell sagt eine Geschwindigkeit unter
  `xai_event_brake_threshold` m/s vorher
- **Warmup-Skip**: Die ersten `xai_skip_first_steps` Steps werden übersprungen
  (Initialisierungsphase wo das Fahrzeug steht)
Wa
Jeder gespeicherte Tensor enthält zusätzlich:
- `save_reason`: Warum gespeichert wurde (`periodic`, `object_nearby_5.2m`, `braking_0.3ms`)
- `pred_target_speed`: Vorhergesagte Zielgeschwindigkeit
- `gt_speed`: Tatsächliche Geschwindigkeit

So kann man gezielt die relevanten Momente analysieren:

```python
import torch
sample = torch.load('xai_tensors/000150.pt')
print(sample['save_reason'])    # z.B. 'object_nearby_3.2m'
print(sample['pred_target_speed'])  # z.B. 4.5  (sollte bremsen!)
print(sample['gt_speed'])       # z.B. 8.3  (fährt noch zu schnell)
```

`xai_save_freq` bestimmt wie oft Tensoren gespeichert werden. Standardmässig alle 100 Steps. Für dichtere Analyse auf z.B. 10 reduzieren (erhöhte Speichernutzung).

---

## Architektur

```
team_code/xai/
  __init__.py            - Public API
  wrapper.py             - Captum-kompatible Wrapper (TF++/LidarCenterNet)
  attributions.py        - Attribution-Engine (XAIEngine Klasse)
  visualization.py       - Heatmap-Rendering (XAIVisualizer Klasse)
  plant_visualization.py - Token-Importance Visualisierung (PlanT2)
  analysis.py            - Standalone Offline-Analyse CLI (TF++ und PlanT2)
```

### Wie die Wrapper funktionieren (TF++)

Captum erwartet Modelle mit einem klaren `forward(input) -> scalar` Interface. Da `LidarCenterNet` 6 Inputs nimmt und 10 Outputs zurückgibt, werden leichtgewichtige Wrapper verwendet:

- **TargetSpeedWrapper** - Isoliert die Geschwindigkeitsvorhersage
- **WaypointWrapper** - Isoliert Waypoint- oder Checkpoint-Vorhersagen
- **BBoxWrapper** - Isoliert die Bounding-Box Confidence
- **SemanticWrapper** - Isoliert semantische Segmentierungs-Logits
- **CaptumForwardAdapter** - Übersetzt zwischen Captums `(inputs, additional_forward_args)` Konvention und dem Modell-Interface

### Multi-Modal Attribution (TF++)

Bei der Attribution werden RGB und LiDAR als primäre Inputs behandelt. Target Point, Geschwindigkeit und Kommando werden als `additional_forward_args` durchgereicht und nicht attributiert:

```
inputs = (rgb, lidar_bev)              <- hierauf werden Gradienten berechnet
additional_forward_args = (target_point, ego_vel, command)  <- konstant gehalten
```

### Token-Level Attribution (PlanT2)

PlanT2 arbeitet nicht mit rohen Sensor-Inputs, sondern mit vorverarbeiteten Objekt-Tokens. Die Attribution beantwortet eine andere Frage:

- **TF++**: "Welche Pixel beeinflussen die Entscheidung?" → Spatial Heatmaps
- **PlanT2**: "Welche erkannten Objekte beeinflussen die Entscheidung?" → Token-Importance-Ranking

```
inputs = (x_objs_features,)            <- Kontinuierliche Token-Features (x,y,yaw,speed,w,l)
additional_forward_args = (x_objs_types,)  <- Typ-Spalte (diskret, konstant gehalten)
```

Die Typ-Spalte (Index 0) wird separat als `additional_forward_args` behandelt, da sie in einer Boolean-Maske (`x_objs[...,0] == i`) verwendet wird und keine sinnvollen Gradienten liefert.

### Attention-Extraktion

**TF++**: Die GPT Fusion-Blöcke im TransfuserBackbone verwenden standardmässig Flash Attention (keine Weights gespeichert). Für XAI kann ein `store_attention`-Flag aktiviert werden, das auf eine explizite Attention-Berechnung umschaltet und die Weights speichert.

**PlanT2**: Der HuggingFace-Transformer gibt Attention-Weights direkt via `output_attentions=True` zurück. Die Self-Attention-Matrizen zeigen, welche Tokens sich gegenseitig beeinflussen.

---

## Interpretation der Ergebnisse

### Heatmaps lesen

- **Rot/Gelb**: Hohe positive Attribution - dieser Bereich treibt die Vorhersage
- **Blau**: Bei "all"-Sign: negative Attribution (drückt Vorhersage in Gegenrichtung)
- **Dunkel/Transparent**: Keine relevante Attribution

### Typische Muster

- **Target Speed "Bremsen"**: Attribution liegt auf Ampeln, Fussgängern, Stopschildern
- **Target Speed "Beschleunigen"**: Attribution liegt auf freier Strasse
- **Checkpoint/Waypoint**: Attribution liegt auf Fahrbahnmarkierungen, Kurvenverläufen
- **LiDAR-dominiert**: Nahbereich-Hindernisse (Fahrzeuge, Maürn)
- **RGB-dominiert**: Ampeln, Verkehrszeichen, entfernte Objekte

### summary.json

Die aggregierten Statistiken zeigen das Verhältnis von RGB- zu LiDAR-Importance:

```json
{
  "saliency": {
    "target_speed": {
      "rgb_importance_mean": 0.62,
      "lidar_importance_mean": 0.38,
      "num_samples": 50
    }
  }
}
```

---

## Beispiele

### Minimales Beispiel: Nur Tensor-Speicherung

```bash
# Evaluation - speichert nur Tensoren, kein XAI-Overhead:
XAI_ENABLED=1 SAVE_PATH=./eval_out \
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --routes ./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml \
  --agent ./team_code/sensor_agent.py \
  --agent-config ./checkpoints/tfpp

# Danach offline analysieren:
cd team_code
python -m xai.analysis \
    --checkpoint ../checkpoints/tfpp \
    --tensor_dir ../eval_out/<route>/xai_tensors \
    --output_dir ../xai_results \
    --methods saliency integrated_gradients \
    --output_heads target_speed checkpoint
```

### Live XAI während Evaluation

```bash
XAI_ENABLED=1 XAI_LIVE=1 XAI_METHOD=saliency XAI_OUTPUT_HEAD=target_speed \
SAVE_PATH=./eval_out DEBUG_CHALLENGE=1 \
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --routes ./fail2drive_split/Generalization_PedestriansOnRoad_1085.xml \
  --agent ./team_code/sensor_agent.py \
  --agent-config ./checkpoints/tfpp
```

### Analyse-Optionen kombinieren

```bash
cd team_code

# Alle Methoden, alle Heads, mit Attention:
python -m xai.analysis \
    --checkpoint ../checkpoints/tfpp \
    --tensor_dir ../eval_out/<route>/xai_tensors \
    --output_dir ../xai_results_full \
    --methods saliency integrated_gradients grad_cam deeplift feature_ablation \
    --output_heads target_speed checkpoint waypoint bbox semantic \
    --attention

# Nur Integrated Gradients für Geschwindigkeit, erste 10 Samples:
python -m xai.analysis \
    --checkpoint ../checkpoints/tfpp \
    --tensor_dir ../eval_out/<route>/xai_tensors \
    --output_dir ../xai_results_speed \
    --methods integrated_gradients \
    --output_heads target_speed \
    --num_samples 10
```

---

## Modality Degradation (A/B Testing)

Mit Modality Degradation kann gezielt eine Sensormodalität (Kamera oder LiDAR) verschlechtert werden, um kausal zu testen, wie abhängig das Modell von jeder Eingabe ist. Die XAI-Analyse läuft anschließend auf den degradierten Daten — so lässt sich beobachten, ob sich die Modality-Importance verschiebt und ob das Modell auf die verbleibende Modalität ausweicht.

### Beispiele

```bash
cd team_code

# Kamerabild unscharf machen (Gaussian Blur):
python -m xai.analysis \
    --checkpoint ../checkpoints/tfpp \
    --tensor_dir ../eval_out/<route>/xai_tensors \
    --output_dir ../xai_results \
    --methods saliency integrated_gradients \
    --output_heads target_speed \
    --degrade rgb --degrade_method blur --degrade_strength 0.5

# LiDAR-Punkte zufällig entfernen (50% Dropout):
python -m xai.analysis \
    --checkpoint ../checkpoints/tfpp \
    --tensor_dir ../eval_out/<route>/xai_tensors \
    --output_dir ../xai_results \
    --methods saliency integrated_gradients \
    --output_heads target_speed \
    --degrade lidar --degrade_method dropout --degrade_strength 0.8

# Kompletter RGB-Ausfall (schwarzes Bild):
python -m xai.analysis \
    --checkpoint ../checkpoints/tfpp \
    --tensor_dir ../eval_out/<route>/xai_tensors \
    --output_dir ../xai_results \
    --methods saliency \
    --output_heads target_speed \
    --degrade rgb --degrade_method zero

# Rauschen auf LiDAR:
python -m xai.analysis \
    --checkpoint ../checkpoints/tfpp \
    --tensor_dir ../eval_out/<route>/xai_tensors \
    --output_dir ../xai_results \
    --methods saliency \
    --output_heads target_speed \
    --degrade lidar --degrade_method noise --degrade_strength 0.3
```

### Parameter

| Flag | Werte | Standard | Beschreibung |
|------|-------|----------|--------------|
| `--degrade` | `rgb`, `lidar` | — | Welche Modalität degradiert wird (nur TF++) |
| `--degrade_method` | `blur`, `noise`, `dropout`, `zero` | `blur` | Art der Degradierung |
| `--degrade_strength` | Float `0.0`–`1.0` | `0.5` | Stärke der Degradierung |

**Methodenkompatibilität:** `blur` ist nur mit `--degrade rgb` verwendbar, `dropout` nur mit `--degrade lidar`. `noise` und `zero` funktionieren mit beiden Modalitäten.

### Output

Die Ergebnisse werden in einem Ordner mit Degradierungs-Info im Namen gespeichert, z.B. `xai_results/20260903_170000_degrade_rgb_blur_0.5/`. Pro Sample wird ein Vergleichsbild `comparison_XXXX.png` erzeugt, das Original und degradierte Eingabe nebeneinander zeigt. Die `summary.json` enthält ein zusätzliches `"degradation"` Feld mit Modalität, Methode und Stärke.

---

## Token Ablation (PlanT2)

Token Ablation ist das PlanT2-Äquivalent zur Modality Degradation bei TF++. Statt Sensordaten zu verschlechtern, werden gezielt Objekte aus der Token-Liste entfernt, um kausal zu testen, welche Objekte die Fahrentscheidung des Modells beeinflussen.

### Beispiele

```bash
cd team_code

# Alle Fussgänger entfernen:
python -m xai.analysis \
    --checkpoint ../checkpoints/plant2 \
    --tensor_dir ../eval_out/<route>/xai_tensors \
    --output_dir ../xai_results \
    --methods saliency integrated_gradients \
    --output_heads target_speed \
    --ablate type --ablate_target walker

# Alle Fahrzeuge entfernen:
python -m xai.analysis \
    --checkpoint ../checkpoints/plant2 \
    --tensor_dir ../eval_out/<route>/xai_tensors \
    --output_dir ../xai_results \
    --methods saliency \
    --output_heads target_speed \
    --ablate type --ablate_target car

# Alle Ampeln entfernen:
python -m xai.analysis \
    --checkpoint ../checkpoints/plant2 \
    --tensor_dir ../eval_out/<route>/xai_tensors \
    --output_dir ../xai_results \
    --methods saliency \
    --output_heads target_speed \
    --ablate type --ablate_target traffic_light

# Nur Objekte innerhalb von 10 Metern behalten:
python -m xai.analysis \
    --checkpoint ../checkpoints/plant2 \
    --tensor_dir ../eval_out/<route>/xai_tensors \
    --output_dir ../xai_results \
    --methods saliency \
    --output_heads target_speed \
    --ablate distance --ablate_target 10.0
```

### Parameter

| Flag | Werte | Beschreibung |
|------|-------|--------------|
| `--ablate` | `type`, `distance` | Ablationsmodus (nur PlanT2) |
| `--ablate_target` | Typname oder Distanz | Ziel: Objekttyp-Name oder Entfernung in Metern |

**Verfügbare Typnamen:** `car`, `walker`, `static`, `stop_sign`, `traffic_light`, `emergency`

### Output

Pro Step wird eine Datei `ablation_info_XXXX.txt` erzeugt, die alle entfernten Tokens auflistet (Typ, Position, Distanz). Die `summary.json` enthält ein `"ablation"` Feld mit Modus, Ziel und Anzahl entfernter Tokens.

### Demo-Skript

```bash
python evaluation/run_plant2_ablation_demo.py
```

Führt automatisch drei Analysen durch (Referenz, ohne Fussgänger, ohne Fahrzeuge) und gibt einen tabellarischen Vergleich aus.

---

## Troubleshooting

| Problem | Lösung |
|---------|---------|
| `No .pt files found` | Evaluation muss mit `XAI_ENABLED=1 SAVE_PATH=...` laufen |
| `CUDA out of memory` bei IG | `--num_samples` reduzieren oder `--methods saliency` verwenden |
| `Model did not produce target speed prediction` | Checkpoint verwendet evtl. andere Config. Prüfe ob `use_controller_input_prediction=Trü` |
| `captum not found` | `pip install captum>=0.7.0` |
| Schwarze Heatmaps | Modell ist im falschen Zustand. Sicherstellen dass der Checkpoint zum config.json passt |

---

---

## PlanT2-Unterstützung

PlanT2 ist ein token-basiertes Modell aus dem [offiziellen Repository](https://github.com/autonomousvision/plant2). Es verwendet privilegierten Zugang zum Simulator (Bounding Boxes statt rohe Sensordaten) und einen HuggingFace-Transformer (BERT-Medium) als Backbone.

### Evaluation starten

```bash
# Standard-Evaluation mit Live-Visualisierung + XAI-Tensor-Speicherung
LIVE_VISU=1 XAI_ENABLED=1 SAVE_PATH=./eval_out \
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --routes ./fail2drive_split/Base_PedestriansOnRoad_0085.xml \
  --agent ./team_code/plant2_agent.py \
  --agent-config ./checkpoints/plant2 \
  --track MAP
```

Wichtig: PlanT2 braucht `--track MAP` (privilegierter Agent).

### Live-Visualisierung

Bei `LIVE_VISU=1` zeigt ein pygame-Fenster:
- **Oben**: RGB-Kamerabild (Frontkamera)
- **Unten**: Synthetisches BEV mit:
  - Farbkodierte Objekt-Tokens (Orange=Auto, Gruen=Fussgaenger, Rot=Ampel, etc.)
  - Geplante Route (blaue Linie)
  - Predicted Path/Waypoints (Farbverlauf)
  - Geschwindigkeitsanzeige (Soll/Ist)

Die Bilder werden gleichzeitig als PNG im Output-Ordner gespeichert.

### XAI-Analyse (Offline)

```bash
cd team_code
python -m xai.analysis \
    --checkpoint ../checkpoints/plant2 \
    --tensor_dir ../eval_out/<route>/xai_tensors \
    --output_dir ../xai_results \
    --methods saliency integrated_gradients \
    --output_heads target_speed checkpoint
```

Das Analyse-Skript erkennt automatisch den Modell-Typ (TF++ oder PlanT2) anhand des Checkpoint-Formats (.pth vs .ckpt).

### PlanT2 Tensor-Format

| Key | Shape | Beschreibung |
|-----|-------|--------------|
| `x_objs` | `(N, 7)` | Alle Objekt-Tokens [type, x, y, yaw_deg, speed_kmh, width, length] |
| `idxs` | `(1, max_seq)` | Indices die x_objs den Batch-Samples zuordnen |
| `route_original` | `(1, 20, 2)` | Geplante Route in Ego-Koordinaten |
| `speed_limit` | `(1,)` | Geschwindigkeitslimit-Kategorie (0-3) |
| `ego_speed` | `(1,)` | Eigengeschwindigkeit |
| `model_type` | `'plant2'` | Identifikator |

### PlanT2 vs. TF++ Vergleich

| Aspekt | TF++ (LidarCenterNet) | PlanT2 (HFLM) |
|--------|----------------------|----------------|
| Input-Typ | Rohe Pixel (RGB + LiDAR BEV) | Objekt-Tokens (privilegiert vom Simulator) |
| Attribution-Level | Pixel-weise Saliency | Token-weise Importance |
| Kernfrage | "Welche Bildregion?" | "Welches Objekt?" |
| Visualisierung | Heatmap auf RGB/BEV | Farbkodierte Objekte + Ranking |
| Attention | Cross-Modal (Image<->LiDAR) | Self-Attention (Token<->Token) |
| XAI-Methoden | Saliency, IG, GradCAM, Feature Ablation, DeepLift | Saliency, IG, Feature Ablation, DeepLift |
| Controller | PID (alter Tuning) | PID (speed-dependent) + Linear Regression |
| Track | `--track SENSORS` | `--track MAP` |

### PlanT2 Dateien

| Datei | Funktion |
|-------|----------|
| `team_code/plant2_agent.py` | Agent fuer CARLA-Evaluation |
| `team_code/plant2_model.py` | HFLM-Netzwerk (Transformer + Waypoint-Decoder) |
| `team_code/plant2_lit_module.py` | PyTorch Lightning Wrapper |
| `team_code/plant2_variables.py` | Konstanten (Objekt-Typen, Speed-Bins) |
| `checkpoints/hf_models/prajjwal1/bert-medium/` | Lokale BERT-Config (kein Internet noetig) |

---

## Evaluation-Toolkit

Das `evaluation/`-Verzeichnis enthält Skripte zur quantitativen Beantwortung der Research Questions. Detaillierte Dokumentation: `evaluation/documentation.md`.

### Schnellstart

```bash
# 1. Evaluationen laufen lassen (CARLA muss laufen)
./evaluation/run_paired_evaluation.sh PedestriansOnRoad

# 2. Quantitativer Vergleich Base vs. Generalization
python evaluation/aggregate_category.py \
    --checkpoint ./checkpoints/tfpp \
    --base_category Base_PedestriansOnRoad \
    --gen_category Generalization_Animals \
    --output_dir ./evaluation/results/rq2

# 3. Kausaler Occlusion-Test (RQ3, nur TF++)
python evaluation/roi_occlusion.py \
    --checkpoint ./checkpoints/tfpp \
    --tensor_dir ./eval_out/Base_ConstructionPermutations_0015/xai_tensors \
    --roi sign 0.35 0.15 0.55 0.45 \
    --roi cones 0.20 0.50 0.80 0.90 \
    --output_dir ./evaluation/results/rq3_occlusion
```

### Verfügbare Evaluations-Skripte

| Skript | Zweck |
|--------|-------|
| `evaluation/paired_comparison.py` | Quantitativer Vergleich zweier Runs (Spatial-Metriken, Modality-Shift) |
| `evaluation/aggregate_category.py` | Aggregiert alle Routen einer Kategorie fuer statistische Vergleiche |
| `evaluation/roi_occlusion.py` | Kausaler Test durch gezieltes Verdecken von Bildregionen (nur TF++) |
| `evaluation/run_paired_evaluation.sh` | Batch-Runner fuer alle Routen einer Szenario-Kategorie |

---

## Modifizierte bestehende Dateien

| Datei | Aenderung |
|-------|-----------|
| `team_code/config.py` | XAI-Konfigurationsblock hinzugefuegt |
| `team_code/requirements.txt` | `captum>=0.7.0` hinzugefuegt |
| `team_code/sensor_agent.py` | Tensor-Speicherung + optionale Live-XAI (TF++) |
| `team_code/transfuser.py` | `store_attention`-Flag fuer Attention-Extraktion |