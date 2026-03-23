# RainScaler (CWA/CorrDiff) – Kurzüberblick

Diese README ist als schneller Einstieg für Betreuung/Supervision gedacht.

## 1) Runs starten
Runs werden im Ordner `methods` gestartet.

- Cluster (SLURM): `sbatch run_rain.sh`, `sbatch run_all4.sh`, `sbatch run_ch1_t2m.sh`, ...
- Lokal/Test: `python rainscaler_rain.py --opt ../deep_learning/options/<config>.json`

Wichtige Entry-Skripte:
- `rainscaler.py` (Basis-Variante)
- `rainscaler_rain.py` (Rain/Single-Channel + Varianten über Config)
- `rainscaler_rain_all4.py` (4 Output-Channels)

## 2) Configs
Alle JSON-Configs liegen unter:
- `../deep_learning/options/`

Typische Beispiele:
- `rainscaler_config_cwb_rain.json`
- `rainscaler_config_cwb_all4.json`
- `rainscaler_config_cwb_ch1_t2m.json`
- `rainscaler_config_cwb_ch2_u10.json`
- `rainscaler_config_cwb_ch3_v10.json`

In den Configs stehen u. a.:
- Datensatz/Channels
- Modellwahl (`model: gan` oder `gan_rain`)
- Loss-Gewichte
- Logging/Plot-Optionen

## 3) Datenloader / Datenschnittstelle
Dataloader-Auswahl:
- `../deep_learning/data/select_dataset.py`

CWA/CorrDiff-Dataset-Wrapper:
- `../deep_learning/data/dataset_cwb_nemo.py`

Die Training-Skripte bauen train/test Loader und Split-Logik direkt in `rainscaler_rain.py` bzw. `rainscaler_rain_all4.py`.

## 4) Modelle
Modell-Dispatch:
- `../deep_learning/models/select_model.py`

Training-Logik:
- GAN-Basis: `../deep_learning/models/model_gan.py`
- Rain-Variante: `../deep_learning/models/model_gan_rain.py`

Netzwerke:
- Generator (Mask/GCN): `../deep_learning/models/network_msrresnet_mask.py`
- Discriminator: `../deep_learning/models/network_discriminator.py`

## 5) Forward und Backprop anschauen:
In `model_gan.py` / `model_gan_rain.py`:
- `feed_data(...)`: Übergabe Batch (LR/HR) an das Modell
- `netG_forward(...)`: Forward durch den Generator
- `optimize_parameters(current_step)`: Loss-Berechnung + Backprop + Optimizer-Step
- `test(...)`: Inferenz/Validierung ohne Gradienten

In den `methods/rainscaler*.py`-Skripten läuft die Epochen-/Step-Schleife und ruft diese Methoden auf.

## 6) Ergebnisse, Logs, Artefakte?
SLURM-Logs:
- `methods/logs/slurm-*.out` / `slurm-*.err`

WandB lokaler Cache:
- `methods/wandb/`

Trainingsartefakte (Modelle, Bilder, Logs):
- `../../res/src_graph/<task_name>/`

Paper/Analyse-Plots (falls aktiviert):
- `../../res/src_graph/<task_name>/paperplots/`

## 7) Präsentationsplots
Die in der Präsentation verwendeten Plot-Exports liegen separat unter:
- `C:/Users/david/PythonProjekte/physicsnemoKIDS/examples/weather/ProjektModelle/David/Plots`
