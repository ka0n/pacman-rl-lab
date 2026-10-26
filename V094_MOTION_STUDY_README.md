# v0.9.4 – Motion-CNN fair von null testen

**Studienfrage:** Hilft der zusätzliche beobachtbare Kanal mit den **Geisterpositionen des unmittelbar vorherigen Schritts**, wenn beide Netze von null auf **derselben** 11×13-Karte trainieren?

## Voraussetzungen

- Vorherigen v0.9.3-Lauf im Vordergrund mit `Strg+C` abbrechen und abwarten, bis die PowerShell-Eingabeaufforderung wieder erscheint. **Nicht** `Stop-CNN-Marathon.ps1` verwenden: Dieses Skript zielt auf den alten Marathon-Controller.
- Es werden **keine** bereits vorhandenen Checkpoints oder v0.9.3-Ausgabeordner verändert.
- v0.9.4 auf v0.9.3 installieren: ZIP enthält `pacman_rl_lab\...`; nach `C:\path\to\` entpacken.
- `python -m pytest -q` testen; Python/Torch/Rust lokal auf Windows verwenden.

## Studienprotokoll

| Bedingung | Kontrolle | Experiment |
|---|---|---|
| Eingabe | 4 sichtbare Spielfeldebenen | 4 + Geisterposition einen Schritt zuvor |
| Architektur | 4,976,708 Parameter | Gleicher Aufbau, zusätzliche Eingangsgewichte |
| Start | zufällig | **gleiche vier gemeinsamen Eingangskanal-Gewichte und alle sonstigen Gewichte**; fünfter Kanal null |
| Training | 1 Mio. Übergänge pro Seed | 1 Mio. Übergänge pro Seed |
| Seeds | 930042, 930043, 930044 | dieselben |
| Geister | 2, Rust, 16 Welten | identisch |
| Reward | Todesstrafe −20, kein Risk-Shaping | identisch |
| Optimizer | Adam, 0.00015, **frisch** | identisch |
| Epsilon | 1.0 → 0.08 (Decay 0.85) | identisch |
| Validierung | alle 25k auf denselben 100 Seeds ab 10000, Auswahl `best.pt` | identisch |
| Finaler Audit | 1000 neue Spiele Seeds 620000–620999 | identisch |

**Bereits abgeschlossen und verifiziert wiederverwenden:** `runs\v092_scratch_death20\best.pt` (CNN-Scratch, Seed 930042). Die Runner-Validierung prüft neben dem ursprünglichen `best.pt` auch den 1-Mio.-Schritt-`last.pt`, Scratch-Provenienz und vollständige Hyperparameter. Damit bleiben **5 Millionen neue Trainingsschritte** übrig (nicht 6 Millionen).

Für die zwei neuen Baseline-Seeds wird von null trainiert, für **alle drei** Motion-Seeds von null mit **funktionsidentischer Initial-Policy** gestartet (zusätzlicher Kanal zunächst null); *keine* Gewichte eines trainierten Netzes werden übertragen. Das isoliert die zusätzliche Bewegungsinformation besser als der frühere Fine-Tuning-Versuch.

**Einschränkung:** Gleiche Seed-Nummer bedeutet gleiche Zufallsinitialisierung des gemeinsamen Netzes und Reproduzierbarkeit des jeweiligen Laufs, aber wegen unterschiedlicher Aktionen nicht zwingend identische Trainings-Episoden. Das Motion-CNN hat geringfügig mehr Parameter. Drei Seeds liefern erste Varianzangaben, aber keinen endgültigen statistischen Beweis. Python/Rust-Parität bleibt unverändert. Die Validierung auf wiederholt denselben 100 Fällen kann die Auswahl überoptimieren, deshalb wird das neue Audit-Seedfenster nur am Ende verwendet.

## Start

```powershell
cd C:\path\to\pacman_rl_lab
.\Start-V094-MotionStudy.ps1 -Plan
.\Start-V094-MotionStudy.ps1
```

Der zweite Befehl startet die vollständige Serie im Vordergrund; das Fenster offen lassen. Andere Arbeit am Rechner ist möglich. Die Ausgaben werden in einem Log unter `runs\v094_motion_experiment_*.log` mitgeschrieben.

Ergebnisse: `runs\v094_motion_scratch\audit_620000.json`, `summary_620000.json`, `study_protocol.json`.

**Bei Abbruch:** Die bis dahin vollständig trainierten Läufe sind weiterhin vorhanden. Den Startbefehl erneut ausführen: fertige Läufe werden nach Checkpoint-Verifikation wiederverwendet. Angefangene Läufe werden *absichtlich nicht* warm-resumed, weil das bisherige Resume Replay-/RNG-Zustände verwirft. Einen angefangenen `runs\v094_motion_scratch\<arch>_seed<seed>`-Ordner manuell umbenennen, bevor die Serie erneut gestartet wird. Das Skript überschreibt ihn niemals.

**Final-Audit nur einmal verwenden.** Die 1000 frischen Test-Seeds sind Teil des Studienprotokolls. Wenn die Ergebnisse neue Hyperparameteränderungen motivieren, für folgende Studien einen anderen Audit-Bereich zurückhalten.
