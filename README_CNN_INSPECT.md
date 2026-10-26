# CNN Inspector (Add-on fuer Pac-Man RL Lab v0.9)

Diese kleine Erweiterung gibt es als `pacman_lab.cnn_inspect`. Sie veraendert **keinen** bestehenden Trainingscode oder Checkpoint, nutzt die bestehende PyTorch/CUDA-/Pygame-Installation und ist kompatibel mit CNN-, CNN-History-, MLP- und GRU-v0.8/v0.9-Checkpoints.

## Installation (Windows, PowerShell)

Das ZIP nach `C:\path\to\` entpacken. Die Ordnerstruktur im ZIP beginnt mit `pacman_rl_lab\`. Die einzige neue Programmdatei ist `pacman_lab\cnn_inspect.py`. **Nicht** `pip install` neu ausfuehren; du hast die Voraussetzungen bereits installiert. Die Originalmodelle unter `runs\` bleiben unangetastet.

Im Ordner `C:\path\to\pacman_rl_lab`:

## A: Dem besten Modell zuschauen

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_inspect watch `
  --checkpoint ".\runs\v09_finetune_cnn_seed42\best.pt" `
  --maze large --seed 80000 --fps 12 --device cuda
```

- **Leertaste** pausieren/fortsetzen; **S** einen Schritt ausfuehren und pausieren.
- **N** naechstes Spiel (naechster Seed); **R** dieselbe Partie wiederholen.
- **+/-** Geschwindigkeit; **D** Q-Werte ein/aus; **Esc** beenden.
- Nach einem Spiel geht es normalerweise nach 1,5 Sekunden weiter. Mit `--pause-on-end` bleibt das Ende bis `N` stehen.
- Es spielen **zwei Geister**, und die Policy waehlt immer die Aktion mit dem hoechsten Q-Wert, **ohne Zufallsaktionen**. Dies entspricht v0.9 `evaluate`.

## B: Die 79 Niederlagen untersuchen

Die Vergleichsevaluation erreichte **92,10 % Siege** auf Seed-Basis 80000, also 79 Niederlagen in 1.000 Spielen. Um *ihre Ursachen* zu sehen, muessen die Partien nochmals detailliert simuliert werden:

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_inspect audit `
  --checkpoint ".\runs\v09_finetune_cnn_seed42\best.pt" `
  --maze large --seed 80000 --episodes 1000 --worlds 16 `
  --engine rust --device cuda `
  --output ".\runs\v09_finetune_loss_diagnosis.json"
```

Der Inspector zeigt direkt die Anzahl Siege, Niederlagen und Gruende. Die JSON-Datei enthaelt pro Partie:

- `reason`: `player_entered_ghost` (Pac-Man betrat ein besetztes Feld), `ghost_moved_or_swapped` (Geist bewegte sich auf Pac-Man zu oder Positionstausch), `timeout` (Schrittlimit) oder `win`.
- `pellet_percent`, `pellets_left`, `steps`, `steps_since_last_pellet`, `wall_bumps`, `stationary_steps`, `max_stall_streak`, `reversals`, `final_player`/`final_ghosts`.
- Listen der verlorenen Seed-Nummern, darunter Partien, die fast gewonnen waren.

**Hinweis:** Hohe Reversal-/Stall-Werte sind Beobachtungen, kein Beweis fuer eine bestimmte interne Ursache. Der Code veraendert die Spielregeln und Reward-Funktion nicht.

## C: Ausschliesslich verlorene Partien anschauen

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_inspect watch `
  --checkpoint ".\runs\v09_finetune_cnn_seed42\best.pt" `
  --maze large --report ".\runs\v09_finetune_loss_diagnosis.json" `
  --fps 10 --device cuda --pause-on-end
```

N springt zum naechsten *verlorenen* Test-Seed. R startet die aktuelle Partie erneut. Am Ende bleibt das Spielfeld stehen, damit du dir Kollision bzw. Zeitlimit ansehen kannst.

## Reproduzierbarkeit

- Das `audit` verwendet dieselben Seed-Folgen und Batch-Reset-Regeln wie das bisherige `cnn_v09 evaluate`. Daher sollten bei 1000 Spielen ab 80000 wieder **92,10 % Siege** erscheinen. Kleine Abweichungen im GPU-Backend waeren zu untersuchen.
- Der Viewer zeichnet das Python-Spiel, der Audit nutzt standardmaessig Rust. Die Regeln sollten dank der vorhandenen Paritaetstests gleich sein; ein identischer Seed ist fuer den Wiedergabevergleich vorgesehen.
- Die Diagnose kennt die terminale Ursache (Kollision oder Zeitlimit), aber noch keine kausale Erklaerung der Strategiefehler wie bewusstes Festhaengen oder schlechte Risikoabschaetzung. Dafuer die verlorenen Partien beobachten.
- `pygame` ist in der normalen Projektinstallation vorhanden. Im Build-Container konnte die Fensterausgabe nicht interaktiv getestet werden; die Diagnose und das Laden der Modelle wurden auf CPU getestet.
