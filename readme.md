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
    --methods saliency integrated_gradients grad_cam \
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
| `XAI_METHOD` | siehe unten | Methode für Live-Modus |
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
  __init__.py         - Public API
  wrapper.py          - Captum-kompatible Modell-Wrapper
  attributions.py     - Attribution-Engine (XAIEngine Klasse)
  visualization.py    - Heatmap-Rendering (XAIVisualizer Klasse)
  analysis.py         - Standalone Offline-Analyse CLI
```

### Wie die Wrapper funktionieren

Captum erwartet Modelle mit einem klaren `forward(input) -> scalar` Interface. Da `LidarCenterNet` 6 Inputs nimmt und 10 Outputs zurückgibt, werden leichtgewichtige Wrapper verwendet:

- **TargetSpeedWrapper** - Isoliert die Geschwindigkeitsvorhersage
- **WaypointWrapper** - Isoliert Waypoint- oder Checkpoint-Vorhersagen
- **BBoxWrapper** - Isoliert die Bounding-Box Confidence
- **SemanticWrapper** - Isoliert semantische Segmentierungs-Logits
- **CaptumForwardAdapter** - übersetzt zwischen Captums `(inputs, additional_forward_args)` Konvention und dem Modell-Interface

### Multi-Modal Attribution

Bei der Attribution werden RGB und LiDAR als primäre Inputs behandelt. Target Point, Geschwindigkeit und Kommando werden als `additional_forward_args` durchgereicht und nicht attributiert:

```
inputs = (rgb, lidar_bev)              <- hierauf werden Gradienten berechnet
additional_forward_args = (target_point, ego_vel, command)  <- konstant gehalten
```

### Attention-Extraktion

Die GPT Fusion-Blöcke im TransfuserBackbone verwenden standardmässig Flash Attention (keine Weights gespeichert). Für XAI kann ein `store_attention`-Flag aktiviert werden, das auf eine explizite Attention-Berechnung umschaltet und die Weights speichert.

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
    --methods saliency integrated_gradients grad_cam feature_ablation \
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

## Troubleshooting

| Problem | Lösung |
|---------|---------|
| `No .pt files found` | Evaluation muss mit `XAI_ENABLED=1 SAVE_PATH=...` laufen |
| `CUDA out of memory` bei IG | `--num_samples` reduzieren oder `--methods saliency` verwenden |
| `Model did not produce target speed prediction` | Checkpoint verwendet evtl. andere Config. Prüfe ob `use_controller_input_prediction=Trü` |
| `captum not found` | `pip install captum>=0.7.0` |
| Schwarze Heatmaps | Modell ist im falschen Zustand. Sicherstellen dass der Checkpoint zum config.json passt |

---

## Modifizierte bestehende Dateien

| Datei | änderung |
|-------|-----------|
| `team_code/config.py` | XAI-Konfigurationsblock hinzugefügt |
| `team_code/requirements.txt` | `captum>=0.7.0` hinzugefügt |
| `team_code/sensor_agent.py` | Tensor-Speicherung + optionale Live-XAI |
| `team_code/transfuser.py` | `store_attention`-Flag für Attention-Extraktion |
