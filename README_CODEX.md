# Entwicklung auf Apple Silicon

Installation und Start: [README.md](README.md). Die aktive Laufzeit ist `.venv/bin/python` (Homebrew-Python 3.13); `scripts/setup-macos.sh` richtet sie und die Rust-Engine ein.

```sh
./scripts/lab.sh doctor
./scripts/lab.sh test
./scripts/lab.sh smoke --device mps --engine both
```

MPS-Prüfungen benötigen echten Metal-Zugriff, der in einer Sandbox fehlen kann. CPU-Tests und MPS-Tests sind getrennt erkennbar. Keine großen Trainingsläufe ohne eine neue Studienentscheidung; Smoke-Tests sind auf kleine Budgets begrenzt.

Der Quellbaum basiert auf v0.9.4. Originalergebnisse: `docs/EXPERIMENT_RESULTS.md` und `docs/summary_620000.json`. Originale Modelle und Trainingsverzeichnisse wurden nicht importiert. Die Analyse darf deren Verhalten nicht aus Zufallsgewichten ableiten. Beobachtungen, Spielregeln, Rewards und Evaluationsseeds nur im Rahmen eines explizit dokumentierten neuen Experiments ändern.

Aktuelle Analyse und Grenzen: [docs/MACOS_MOTION_AUDIT.md](docs/MACOS_MOTION_AUDIT.md). Für `cnn_motion` bleibt die genaue Ursache des historischen Qualitätsverlusts offen, solange die Originalgewichte und Trainingskurven fehlen.
