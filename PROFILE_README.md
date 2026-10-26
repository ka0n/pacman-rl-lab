# Pipeline-Profiling auf Apple Silicon

```sh
./scripts/lab.sh profile --architecture cnn_single --device cpu --engine python --output /tmp/profile_single_cpu_python.json
./scripts/lab.sh profile --architecture cnn_single --device cpu --engine rust --output /tmp/profile_single_cpu_rust.json
./scripts/lab.sh profile --architecture cnn_single --device mps --engine python --output /tmp/profile_single_mps_python.json
./scripts/lab.sh profile --architecture cnn_single --device mps --engine rust --output /tmp/profile_single_mps_rust.json
.venv/bin/python -m pacman_lab.profile_report --folder /tmp
```

Der Launcher begrenzt seine Standardwerte auf 64 Übergänge, Warmup 16 und Batchgröße 4. Explizite CLI-Optionen können diese Werte überschreiben. Der Profiler führt echte Optimizer-Schritte an frischen Modellen aus und speichert keine Gewichte. Der normale Python-Modulaufruf hat weiterhin ein Budget von 3000 Schritten; für kurze Prüfungen den Launcher verwenden oder `--steps` setzen.

MPS und CUDA werden pro Abschnitt synchronisiert, wenn nicht `--no-gpu-sync` gewählt wird. Dadurch entstehen messbare Zusatzkosten. `gpu_sync_each_section` beschreibt beide Backends, das historische Feld `cuda_sync_each_section` nur CUDA. Die Berichtstabelle nennt Gerät und Engine; Python/Rust-Verhältnisse werden ausschließlich innerhalb desselben Geräts gebildet. Nicht verfügbare Peak-Memory-Messungen werden als `n/a` dargestellt.

Das ist der Einzelwelt-Pfad. Der begrenzte Batch-Smoke-Test läuft über `./scripts/lab.sh smoke`. Reine Simulatorleistung misst `./scripts/lab.sh bench`. Kurze Profile, erste Metal-Kernel-Kompilierung und zwei Validierungsepisoden ergeben keine Aussage zur Lernqualität oder zur dauerhaften Trainingsgeschwindigkeit.
