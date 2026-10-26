# Pac-Man RL Lab v0.9.2 – Safety-Fine-Tuning & Gefahrenanalyse

**Nur Patch** für bestehende v0.9 mit installierter CNN-Inspector-Erweiterung. ZIP nach
`C:\path\to\` entpacken (sie enthält `pacman_rl_lab\...`). Bereits
trainierte `runs/`, `best.pt` und `last.pt` werden nicht verändert.

## 1. Die 79 bisherigen Niederlagen genauer untersuchen

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_inspect audit `
  --checkpoint ".\runs\v09_finetune_cnn_seed42\best.pt" `
  --maze large --seed 80000 --episodes 1000 `
  --worlds 16 --engine rust --device cuda `
  --risk-analysis `
  --output ".\runs\v092_loss_risks_80000.json"
```

Zusätzlich zu `INSPECT` werden Zählungen über die letzten fünf **Entscheidungen
vor dem Episodenende** ausgegeben, getrennt nach Sieg und Niederlage:

- `two_ghosts_near_last_5`: zwei Geister höchstens 5 Schritte entlang der Wege entfernt.
- `opposite_approach_last_5`: zwei nahe Geister haben keine gemeinsame erste
  Richtung auf einem kürzesten Weg zum Spieler – Näherung für eine Zange.
- `blind_branch_threat_last_5`: Pac-Man war in einem echten blind endenden
  Gangabschnitt und ein Geist höchstens 5 Wegschritte entfernt.
- `constrained_exits_last_5`: mindestens zwei geometrische Abgänge, davon
  höchstens einer außerhalb der nächsten möglichen Geisterbewegung.

**Diese Merkmale sind korrelativ, können sich überlappen und beweisen keine
Unvermeidbarkeit.** Die Geisterrichtung wird nicht direkt beobachtet, und zwei
Geister auf demselben Feld können im binären Beobachtungskanal zusammenfallen.
Die Ereignisse werden aus dem **Zustand vor der Entscheidung** ermittelt. Die
simulierte Bewegungslogik und die bisherige greedy-Evaluation ändern sich nicht.

Die JSON-Datei liefert pro Spiel `risk.last_5`, `risk.last_step` und `risk.exposure_steps`
mit konkreten Positionen/Wegdistanzen. Verlustspiele weiter wie bisher im
Inspector mit `watch --report ...` ansehen.

## 2. Kontrolliertes Training vom bewährten 92,1-%-Checkpoint

Zwei neue Läufe starten unabhängig vom **selben** Originalcheckpoint, mit
identischen Start-Seeds, Lernrate, Exploration und Schrittzahl. Nicht mit
`--resume` eines älteren Laufs starten.

**A – nur Todesstrafe -20 (statt -5)**:

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_v09 finetune `
  --checkpoint ".\runs\v09_finetune_cnn_seed42\best.pt" `
  --output ".\runs\v092_safety_death20" `
  --steps 300000 --worlds 16 --engine rust --device cuda `
  --learning-rate 0.00001 `
  --epsilon-start 0.07 --epsilon-end 0.03 `
  --eval-every 25000 --eval-episodes 100 --seed 920042 `
  --death-penalty -20
```

**B – dieselbe Todesstrafe und optionale topologische Risikobewertung**:

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_v09 finetune `
  --checkpoint ".\runs\v09_finetune_cnn_seed42\best.pt" `
  --output ".\runs\v092_safety_topology" `
  --steps 300000 --worlds 16 --engine rust --device cuda `
  --learning-rate 0.00001 `
  --epsilon-start 0.07 --epsilon-end 0.03 `
  --eval-every 25000 --eval-episodes 100 --seed 920042 `
  --death-penalty -20 --risk-shaping-weight 0.5
```

`death_penalty` betrifft **nur einen tatsächlichen Geisttod**, nicht Siege oder
Zeitlimits. Die Anpassung erfolgt auf den Lern-Rewards *nach* dem Simulator-
Schritt; Python und Rust behalten genau dieselben Spielregeln. Bei B wird
zusätzlich ein Potential auf Basis von Wegdistanzen, Geisterpaaren, Sackgassen
und verbleibenden Ausgängen verwendet. Der Potential-Unterschied ist
`0.99 * phi(neu) - phi(alt)` mit `phi(terminal)=0`. Diese Heuristik gibt **keine
festen Aktionen** vor und beeinflusst keine greedy-Evaluation.

Der alte Checkpoint und frühere Ergebnisse bleiben erhalten. `best.pt` der
neuen Läufe wird wie bisher anhand von Siegesrate, dann Pellet-Prozent auf
Validierungsspielen (Seed 10000) gewählt. Optimizer wird übernommen, Replay
neu aufgebaut. Die Sicherheitsparameter werden in `config.json` und neuen
`last.pt`/`best.pt` gespeichert, damit ein Resume sie nicht vergisst.

## 3. Fairer Vergleich auf 1.000 neuen Spielen

Den Vergleich erst nach beiden Läufen durchführen und **nicht** die Test-Seeds
für erneute Modellauswahl optimieren:

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_v09 evaluate `
  --checkpoint ".\runs\v09_finetune_cnn_seed42\best.pt" `
  --checkpoint ".\runs\v092_safety_death20\best.pt" `
  --checkpoint ".\runs\v092_safety_topology\best.pt" `
  --maze large --episodes 1000 --test-seed 120000 `
  --worlds 16 --engine rust --device cuda `
  --output ".\runs\v092_safety_comparison_120000.json"
```

Für eine detaillierte Auswertung des besten neuen Netzes `cnn_inspect audit`
mit `--risk-analysis` und `--seed 120000` aufrufen. Dafür einen **neuen**
`--output`-Dateinamen benutzen: vorhandene Audit-Dateien werden zum Schutz
vor versehentlichem Überschreiben abgelehnt.

## 4. Abgebrochenen Safety-Lauf fortsetzen

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_v09 resume `
  --checkpoint ".\runs\v092_safety_death20\last.pt" `
  --output ".\runs\v092_safety_death20" --device cuda
```

`resume` reinitialisiert Replay und Welten; Gewichte/Optimizer/Zeitplan werden
gemäß dem gespeicherten Run fortgeführt. Gegebenenfalls den Pfad auf Variante B
ändern.

## Ziel / Caveats

Die Ausgangsbasis waren 921/1000 Siege auf Seed 80000 und 79 Geistkontakte.
Ziel ist **weniger Niederlagen**, ohne dass Pac-Man weniger Punkte einsammelt.
Die eigentliche Wirksamkeit von Safety-Shaping und Todesstrafe ist noch nicht
nachgewiesen: erst der Vergleich auf 120000 entscheidet.
