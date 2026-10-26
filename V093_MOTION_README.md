# Pac-Man RL Lab v0.9.3 – Geisterbewegung und frühe Einkesselungsrisiken

Dieses Update ist ein **Patch** auf dein installiertes v0.9.2 (inkl. Inspector).
Es überschreibt nur Quellcode, Dokumentation und Tests, **keine** Checkpoints,
keine `runs`-Ordner und keine gespeicherten Reports. ZIP im Verzeichnis
`C:\path\to\` entpacken; die ZIP enthält `pacman_rl_lab\...`.
Danach in PowerShell im Projektordner bleiben.

## Was geändert wurde

- `cnn_motion`: CNN mit fünf Eingangs-Ebenen: aktuelle Wände, Punkte, Pac-Man,
  Geister plus **Geisterbelegung im vorangegangenen Schritt**. Das ist ein
  ausschließlich aus sichtbaren Spielbildern abgeleitetes Bewegungssignal;
  es kennt weder die wahren Ghost-IDs noch deren internen Richtungszustand.
  Bei Episodenstart wird die aktuelle Ebene wiederholt (keine erfundene Bewegung).
- Gewichtsübertragung `cnn_single` -> `cnn_motion`: Alle vorhandenen Gewichte
  werden übernommen; die Gewichte für den **fünften Kanal** der ersten Faltung
  werden nullinitialisiert. Das neue Netz trifft vor dem ersten zusätzlichen
  Trainingsschritt dieselben Entscheidungen wie das Ausgangs-CNN. Bei der
  Übertragung wird der Optimizer neu initialisiert (Parameterform geändert).
- `--fresh-optimizer` für die gleich behandelte Kontrollgruppe `cnn_single`.
  Alle Trainingsläufe schreiben in getrennte Verzeichnisse.
- `cnn_inspect audit --risk-analysis` speichert jetzt auch **drei Zeitfenster**
  relativ zum Episodenende: 11–20, 6–10 und 1–5 Aktionen davor; dabei wird
  getrennt nach Siegen und Niederlagen gezählt. Eine nicht verfügbare
  Frühphase bei zu kurzen Partien wird aus der jeweiligen Zählung ausgeschlossen.
  Diese retrospektive Statistik ist **keine Vorhersage und kein Beweis für
  vermeidbare Todesfälle**.
- Wiedergabe und v0.9-Audit funktionieren auch mit `cnn_motion` Checkpoints.
  Die Spielmechanik und der Reward-Code in Python/Rust wurden nicht geändert.

## 1. Installieren und Tests ausführen

```powershell
# ZIP nach C:\path\to\ entpacken, dann:
cd C:\path\to\pacman_rl_lab
.\.venv\Scripts\python.exe -m pytest -q
```

## 2. Erst die bisherigen Todesfälle früher untersuchen (kostet kein Training)

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_inspect audit `
  --checkpoint ".\runs\v092_safety_death20\best.pt" `
  --maze large --seed 220000 --episodes 1000 `
  --worlds 16 --engine rust --device cuda --risk-analysis `
  --output ".\runs\v093_baseline_risks_220000.json"
```

Sieh in der Konsole auf `early_11_20`, `middle_6_10` und `terminal_1_5`,
insbesondere auf die **Anteile** pro gewonnener bzw. verlorener Partie.
Die Rohzahlen sind wegen 936+ gewonnener vs. 64 verlorener Episoden nicht
unmittelbar vergleichbar. Die neuen Risikoflags charakterisieren bereits
bekannte Situationen, nicht die tatsächliche zukünftige Ghost-Bewegung.

## 3. Der kontrollierte Trainingstest – zwei unabhängige 400k-Schritte-Läufe

Beide Läufe starten vom bisher besten Safety-CNN und benutzen Todesstrafe −20.
**Zuerst die Kontrollgruppe** (unveränderte vier Ebenen), aber mit *frischem*
Adam-Optimizer, genau wie der Motion-Transfer:

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_v09 finetune `
  --checkpoint ".\runs\v092_safety_death20\best.pt" `
  --architecture cnn_single --fresh-optimizer `
  --output ".\runs\v093_control_cnn" `
  --steps 400000 --worlds 16 --engine rust --device cuda `
  --learning-rate 0.00002 `
  --epsilon-start 0.12 --epsilon-end 0.04 `
  --eval-every 25000 --eval-episodes 100 --seed 930042 `
  --death-penalty -20
```

**Danach die Motion-Variante**, gleicher Startcheckpoint und dieselben
Trainingsparameter; einzig die zusätzliche Geisterhistorie ist neu:

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_v09 finetune `
  --checkpoint ".\runs\v092_safety_death20\best.pt" `
  --architecture cnn_motion `
  --output ".\runs\v093_motion_cnn" `
  --steps 400000 --worlds 16 --engine rust --device cuda `
  --learning-rate 0.00002 `
  --epsilon-start 0.12 --epsilon-end 0.04 `
  --eval-every 25000 --eval-episodes 100 --seed 930042 `
  --death-penalty -20
```

**Wichtig:** Die Quell-Checkpoint-Dateien bleiben erhalten. Die beiden Läufe
sind von Anfang an getrennt und werden nicht gegeneinander fortgesetzt. Die
`best.pt`-Datei wird auf 100 festen Validierungsspielen ausgewählt; die
Zwischenergebnisse sind deshalb optimistischer als frische Tests. Standard-
Checkpoint-`resume` über `python -m pacman_lab.cnn_v09 resume` bleibt möglich.

## 4. Fairer Vergleich auf 1.000 neuen Partien

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_v09 evaluate `
  --checkpoint ".\runs\v092_safety_death20\best.pt" `
  --checkpoint ".\runs\v093_control_cnn\best.pt" `
  --checkpoint ".\runs\v093_motion_cnn\best.pt" `
  --maze large --episodes 1000 --test-seed 240000 `
  --worlds 16 --engine rust --device cuda `
  --output ".\runs\v093_motion_comparison_240000.json"
```

Die drei Netze spielen **dieselben** 1.000 Partien. Nur dann kann man beurteilen,
ob die Geisterhistorie gegenüber dem normalen CNN mit *derselben zusätzlichen
Trainingszeit* hilft. Im Zweifelsfall die gewonnenen Spiele je Seed paarweise
betrachten, nicht allein die gerundete Siegesrate.

## 5. Mit dem neuen Modell zuschauen / Gründe vergleichen

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.cnn_inspect watch `
  --checkpoint ".\runs\v093_motion_cnn\best.pt" `
  --maze large --seed 240000 --fps 10 --device cuda

.\.venv\Scripts\python.exe -m pacman_lab.cnn_inspect audit `
  --checkpoint ".\runs\v093_motion_cnn\best.pt" `
  --maze large --seed 240000 --episodes 1000 `
  --worlds 16 --engine rust --device cuda --risk-analysis `
  --output ".\runs\v093_motion_risks_240000.json"
```

**Grenzen:** Das ist ein Test der **Geisterbewegungsinformation**, keine
perfekte Vorausplanung. Wird damit die Siegesrate nicht verbessert, könnten
wir im nächsten Schritt einen kurzen Forward-Rollout über mögliche
Geisterzüge testen. Die GRU wird in diesem Patch nicht verändert. Rust/CUDA
ist auf deinem Windows-PC zu testen; die Tests in der Erstellungsumgebung
laufen mit Python/CPU.
