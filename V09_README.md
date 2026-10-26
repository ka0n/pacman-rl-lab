# v0.9: Held-out-Evaluation und sicheres Fine-Tuning aus v0.8-Checkpoints

Diese Version ergänzt v0.8. **Die Spielmechanik, CNN-Architektur und Reward-Funktion sind unverändert.** V0.8-Checkpoints (`best.pt`, `last.pt`) bleiben kompatibel. Neue Befehle unter `python -m pacman_lab.cnn_v09`.

## Sichere Installation unter Windows

**Nur installieren, wenn der bisherige Marathon gestoppt ist.** Quellcode nicht während eines laufenden Jobs überschreiben. Vorher alle `best.pt` / `last.pt` und den Suite-Ordner behalten, möglichst zusätzlich sichern.

1. Marathon regulär nach `cnn_gru_seed1042` stoppen (`.\Status-CNN-Marathon.ps1`; `.\Stop-CNN-Marathon.ps1 -Stop`). Der dritte Seed muss nicht durchgeführt werden.
2. ZIP `pacman_rl_lab_windows_v0.9_finetune.zip` nach `C:\path\to\` entpacken, bestehenden `pacman_rl_lab`-Ordner aktualisieren. Die ZIP enthält **keine** `runs`, `.venv` oder kompilierte Rust-DLL; diese bleiben erhalten.
3. PowerShell:

```powershell
cd C:\path\to\pacman_rl_lab
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Die bestehende Rust-DLL muss unter `native\pacman_native\target\release\pacman_native.dll` liegen. Bei einer frischen Installation `cargo build --release` im Rust-Unterordner ausführen. Die v0.8-Rust-Engine wurde nicht geändert.

## 1. ZUERST: Baseline auf neuen Testfällen (ohne Training)

Dieser Befehl wertet **alle verfügbaren** `best.pt` der Architektur `cnn_single` und `cnn_history4` aus dem alten Marathon aus – inklusive Seed 42 und 1042. 1.000 identische, neue Episode-Seeds auf der bekannten und zwei bislang ungesehenen Topologien pro Checkpoint und Map.

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_v09 evaluate `
  --suite ".\runs\cnn_20261013_112458" `
  --episodes 1000 --worlds 16 --engine rust --device cuda `
  --output ".\runs\v09_baseline_audit.json"
```

Alle Modelle bekommen exakt dieselben Test-Seeds `70000..70999`. Das Ergebnis enthält Win-%, Pellet-%, Spieldauer, Verhalten, Checkpoint-Hash und Modellherkunft. **Die Audit-Testdaten sollten nicht für die fortlaufende Auswahl von Hyperparametern/Checkpoints verwendet werden**, sonst sind sie nicht länger unabhängig. Die v0.8-Best-Modelle wurden bereits mit separaten Validierungs-Episoden ausgewählt.

Für einen schnellen Funktionstest `--episodes 10` und neuen Ausgabedateinamen benutzen. `--engine python` ist eine langsame Fallback-Option ohne Rust-DLL.

## 2. Fine-Tuning vom besten vorhandenen Checkpoint (separater Ordner)

Beispiel CNN Seed 1042 (`best.pt` erreichte in der v0.8-Validierung 95 % Siege); lädt CNN-Gewichte + Adam-Zustand und senkt die Lernrate, startet aber Replay-Buffer, Explorations-Schedule, Trainings-RNG, Welten und Evaluationshistorie neu. Geplante **zusätzliche 500.000** Trainingsübergänge auf derselben Karte, nicht 500.000 inklusive der früheren Million.

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_v09 finetune `
  --checkpoint ".\runs\cnn_20261013_112458\cnn_single_seed1042\best.pt" `
  --output ".\runs\v09_finetune_cnn_seed1042" `
  --steps 500000 --worlds 16 --engine rust --device cuda `
  --learning-rate 0.00003 --epsilon-start 0.10 --epsilon-end 0.05 `
  --eval-every 25000 --eval-episodes 100
```

Neue Dateien `runs\v09_finetune_cnn_seed1042\best.pt`, `last.pt`, `fine_tune.json`, `config.json`, `metrics.csv`. Der **Originalcheckpoint bleibt unangetastet**; neue `step`-Angaben zählen nur die zusätzlichen Schritte. `best.pt` der Fine-Tune-Runde ist das beste **innerhalb der neuen Runde** auf deren Validierungs-Seeds, nicht zwangsläufig besser als die ursprüngliche Quelle. Vergleiche das mit dem unabhängigen Audit. Keine Garantie für eine Verbesserung.

**Unterbrochenes Fine-Tuning fortsetzen:**

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_v09 resume `
  --checkpoint ".\runs\v09_finetune_cnn_seed1042\last.pt" `
  --output ".\runs\v09_finetune_cnn_seed1042" --device cuda
```

Das ist wie v0.8 ein **Warm-Restart**, kein exakter Replay: Replay-Buffer, Zufallszustände und laufende Episoden werden zurückgesetzt. `--resume` kann die Lernkurve deshalb beeinflussen.

## 3. Erneuter unabhängiger Vergleich nach dem Fine-Tuning

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_v09 evaluate `
  --checkpoint ".\runs\cnn_20261013_112458\cnn_single_seed1042\best.pt" `
  --checkpoint ".\runs\v09_finetune_cnn_seed1042\best.pt" `
  --episodes 1000 --worlds 16 --engine rust --device cuda `
  --output ".\runs\v09_before_after_audit.json"
```

Diese Gegenüberstellung ist paarweise auf identischen Seeds. Wegen der zuvor betrachteten Baseline-Audit-Seeds ist der zweite Durchlauf ein **vergleichbarer Nachtest**, aber nicht mehr ein völlig neuer, blind gehaltener Test. Für einen zusätzlichen frischen Test `--test-seed 80000` und neuen Dateinamen wählen.

### Grenzen

- Fine-Tuning kann eine bereits gute Policy verschlechtern. Alter `best.pt` bleibt erhalten.
- Die GRU wird in v0.9 **noch nicht repariert**; das ist ein separates Diagnoseexperiment.
- Die Ergebnisse aus dem früheren 40-Episoden-Validierungslauf haben deutliche Stichprobenstreuung; große Held-out-Mengen sind für fundierte Aussagen wichtig.
- Die Testmöglichkeiten umfassen die in v0.8 definierten 11×13-Layouts. Für tatsächlich neue Kartengrößen benötigt das aktuelle CNN wegen der Flatten/Linear-Schicht eine Modellanpassung.
- Auf Windows GTX 1080 Ti/CUDA nicht lokal ausgeführt: CPU-Tests wurden in dieser Build-Umgebung ausgeführt; echte GPU-/Rust-Leistung auf Ziel-PC prüfen.
