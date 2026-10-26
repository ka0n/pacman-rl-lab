# macOS-Port und Untersuchung des Motion-CNN

Die Apple-M5-Version läuft mit PyTorch/MPS und der nativen arm64-Rust-Engine. Im untersuchten Beobachtungs-, Replay- und Double-DQN-Pfad wurde kein nachgewiesener Fehler gefunden, der den historischen Motion-Nachteil erklärt. Zwei Fehler im Studien-Runner wurden reproduziert und behoben. Die fehlenden Originalgewichte und Trainingskurven begrenzen die Ursachenanalyse.

Es wurden ausschließlich Unit-Tests, begrenzte Smoke-Tests und kurze Benchmarks ausgeführt. Karten, Rewards, Spielregeln, Observation-Semantik und historische Evaluationsseeds wurden nicht verändert. Die Windows-Experimente wurden nicht neu ausgewertet.

## 1. Historische Evidenz

Quelle: unveränderte `EXPERIMENT_RESULTS.md` und `summary_620000.json` in diesem Ordner. Je Modell und Trainingsseed wurden 1000 gemeinsame Held-out-Episoden ausgewertet.

| Trainingsseed | CNN Siege % | Motion Siege % | Motion minus CNN, pp |
|---|---:|---:|---:|
| 930042 | 89,3 | 75,4 | −13,9 |
| 930043 | 90,1 | 76,2 | −13,9 |
| 930044 | 89,2 | 77,1 | −12,1 |
| Mittel | **89,53** | **76,23** | **−13,30** |

Mittlere Pelletquote: 97,1969 % gegenüber 93,5177 %. Der gleichgerichtete Unterschied über drei Seeds ist ein ernstzunehmender Befund. Er identifiziert weder einen Codefehler noch einen bestimmten Optimierungsmechanismus. Drei Seeds liefern weiterhin nur begrenzte Varianzschätzungen.

Die Originalmodelle, Replay-Inhalte, vollständigen Lernkurven und Laufverzeichnisse sind hier nicht verfügbar. Deshalb kann diese Untersuchung weder die ursprünglichen Q-Werte noch Fehlentscheidungen dieser Policies rekonstruieren. Die Smoke-Modelle sind frische Zufallsmodelle und ersetzen diese Evidenz nicht.

## 2. Beobachtung und zeitliche Ausrichtung

Geprüft: `pacman_lab/core.py`, `native.py`, `cnn_motion.py`, `memory.py`, `cnn_batch.py`, `cnn_train.py`, `cnn_inspect.py` und die Rust-Implementierung.

Mit `G_t` als binärer Geisterbelegung gilt:

```text
cnn_single: s_t = [W_t, Pellets_t, Player_t, G_t]
cnn_motion: s_t = [W_t, Pellets_t, Player_t, G_t, G_(t-1)]
Reset:      s_0 = [W_0, Pellets_0, Player_0, G_0, G_0]
Übergang:   s_(t+1)[4] == s_t[3]
```

Die Koordinaten sind absolute Rasterkoordinaten. Es erfolgt keine Verschiebung relativ zur Spielerposition. Deshalb muss die alte Geisterebene auch nicht um die Spielerbewegung korrigiert werden. Die vier aktuellen Ebenen kommen vollständig aus demselben Simulatorzustand; nur Ebene 5 stammt aus dem vorherigen beobachteten Zustand.

| Prüfpunkt | Ergebnis und Absicherung |
|---|---|
| Reihenfolge und Skalierung | Wände, Pellets, Spieler, aktuelle Geister; beim Motion-CNN zusätzlich vorherige Geister. Replay speichert `uint8` mit 0/1, das Netz erhält `float32` mit 0/1. Keine fehlerhafte RGB-/255-Normalisierung. |
| Zeitlicher Versatz | `GhostMotionHistory.step` kopiert die alte Geisterebene, bevor der neue Frame übernommen wird. Tests über mehrere Schritte und echte Spiele bestätigen genau einen Schritt Historie. |
| Reset | Aktuelle Belegung wird wiederholt. Keine Geisterreste aus der vorigen Episode; keine erfundene Anfangsgeschwindigkeit. |
| Eigentum an Arrays | Eingaben werden intern kopiert; Ausgaben sind eigene Arrays. Änderungen am Simulatorpuffer oder an zurückgegebenen Zuständen verändern weder Historie noch alte Replay-Einträge. |
| Parallele Welten | Jede Welt besitzt einen eigenen Observer. Versetzte Resets und unterschiedliche Episodenenden mischen keine Historien. |
| Training/Evaluation/Viewer | Dieselbe Observer-Auswahl; `current_frame` nimmt beim Motion-CNN die ersten vier Ebenen. Einzel- und Batch-Evaluation werden bereits durch bestehende Paritätstests verglichen. |
| Python/Rust | Beide erzeugen dieselben vier semantischen Ebenen. Der Motion-Observer sitzt danach in Python. Neue Tests vergleichen auch die fünf Ebenen bei unterschiedlichen Bewegungstakten und einzelnen Welt-Resets. |

Die Geisterhistorie wird auch nach explorativen Aktionen fortgeschrieben. Es gibt keinen History-Update nur im Greedy-Zweig. Nach einem tödlichen Spielerzug können die Geister im terminalen Frame unverändert sein, weil die Simulation dann beendet wird. Das entspricht der bestehenden Spielregel und ist kein zeitlicher Fehler.

## 3. Replay und Lernimplementierung

Für beide Feedforward-CNNs wird dieselbe `deque(maxlen=replay_capacity)` verwendet. Jeder Eintrag enthält vollständig kopierte `(s_t, a_t, r_t, s_(t+1), done)`-Daten. Die Geisterhistorie ist bereits Teil des Eintrags; sie wird beim zufälligen Sampling nicht aus benachbarten Buffer-Einträgen rekonstruiert. Zufälliges Sampling zerstört deshalb keine zeitliche Zuordnung.

Der echte Batch-Trainer wird im neuen Integrationstest mit drei kontrollierten Welten durchlaufen: Tod, Gewinn und Timeout sowie ein letztes unvollständiges Batch-Budget. Die Optimierung ist dort ausdrücklich gesperrt. Der Test prüft Terminalbeobachtungen vor dem Reset, Startzustände danach, Reward-Anpassung −20 nur bei Tod, fortlaufende Reset-Seeds, Kopien, FIFO-Kapazität und genau sieben gespeicherte Übergänge.

Das verwendete Double-DQN-Ziel ist:

```text
a* = argmax_a Q_online(s_next, a)
y  = reward + (1 - done) * 0.99 * Q_target(s_next, a*)
```

Ein analytischer Tabellennetz-Test lässt Online- und Target-Argmax absichtlich auseinanderfallen. Der berechnete Huber-Loss stimmt mit dem richtigen Double-DQN-Ziel überein. Kein Gradient läuft durch Target-Werte oder die Berechnung des nächsten Zustands. `done` unterdrückt das Bootstrap korrekt.

`done = terminated OR truncated` behandelt auch Zeitlimits als terminal. Das ist eine bestehende Modellierungsentscheidung beider Modelle; sie wurde nicht stillschweigend auf eine andere Time-limit-Semantik umgestellt. Der Replay-Test unterscheidet Timeout und Tod ausdrücklich.

Adam, Lernrate, Warmup, Batchgröße, Updatefrequenz und Target-Kopien sind für die beiden Bedingungen gleich. Bei 16 Welten und `train_every=8` entstehen nach Warmup zwei Updates je Vector-Step, nicht nur eines. CNN und Motion-CNN benutzen denselben Codezweig. Die Modelle enthalten weder BatchNorm noch Dropout, die einen versteckten Train-/Eval-Unterschied verursachen würden.

Die matched-scratch-Initialisierung wird jetzt über den echten Trainer für alle drei Studienseeds geprüft, ohne Optimizer-Schritte: gemeinsame Gewichte sind bitgleich, Kanal 5 ist null, Target und Online stimmen überein, Adam ist frisch, `fine_tune` bleibt leer. Anfangs-Q-Werte bleiben bei beliebiger fünfter Ebene gleich.

Resume bleibt ein **Warm-Restart**: Modell, Target und Optimizer werden restauriert, Replay, RNG und Welten jedoch nicht. Das ist keine exakte Fortsetzung. Der Studien-Runner lehnt partielle Studienordner weiterhin ab. Metadatenprüfung allein kann nachträglich keinen vollständigen lückenlosen Trainingsverlauf beweisen.

## 4. Nachgewiesene und behobene Fehler

### Protokoll kann nach Speichern nicht wiederverwendet werden

`cnn_motion_study.main` verglich eingelesene JSON-Daten mit `SEEDS` als Python-Tupel. JSON wandelt dieses Tupel in eine Liste um. Ein unverändertes, gespeichertes Protokoll löste deshalb beim nächsten Aufruf einen vermeintlichen Konfigurationswechsel aus.

Korrektur: Auch das erwartete Protokoll verwendet `list(SEEDS)`. Bereits vorhandene JSON-Protokolle bleiben damit lesbar. Der Regressionstest startet den Runner zweimal, unterbricht ihn jeweils vor jedem tatsächlichen Training und prüft anschließend, dass eine echte Lernratenänderung weiterhin abgelehnt wird.

Dieser Fehler betrifft Wiederanläufe, nicht die Qualität eines in einem Aufruf abgeschlossenen Trainings.

### Unzureichender Abschlussnachweis durch `last.pt`

`verify_checkpoint` prüfte beim ausgewählten `best.pt` die Studienparameter, beim ergänzenden `last.pt` aber nur `step == 1_000_000`. Ein fremder Seed, eine andere Architektur oder ein Fine-Tuning-Checkpoint konnte deshalb einen scheinbar gültigen Abschluss belegen.

Korrektur: Der Abschlusscheckpoint durchläuft dieselbe vollständige Prüfung wie der ausgewählte Checkpoint. Zusätzlich werden Architektur im Payload und in der Konfiguration abgeglichen und Checkpoint-Schritte außerhalb des Studienbudgets abgelehnt. Regressionstests decken fremden Seed, Fine-Tuning-Herkunft, andere Architektur, Schritt 0, Budgetüberschreitung und widersprüchliche Metadaten ab.

Das beseitigt eine Integritätslücke. Es gibt keine Evidenz, dass die ursprünglichen sechs Ergebnisse dadurch kontaminiert waren.

### Fehlerhafte Vergleichsmöglichkeit in gemischten Geräteprofilen

Der Profilbericht gruppierte nur nach Architektur und Engine. Bei gemischten CPU-/MPS-Dateien konnte er eine CPU/Python-Messung mit MPS/Rust vergleichen. Die Gruppierung berücksichtigt jetzt auch das Gerät; die Tabelle weist es aus. Der Regressionstest verbietet ausdrücklich einen solchen geräteübergreifenden Rust/Python-Quotienten.

Die Synchronisierung des Profilers unterstützte MPS bereits. Neu sind zutreffende Metadaten (`gpu_sync_each_section`), eine MPS-Bezeichnung und `n/a` statt erfundener Nullwerte bei nicht gemessenem Peak-Speicher.

## 5. Kritische Architekturbewertung

Das Motion-Netz kann die Single-CNN-Funktion exakt darstellen: vier Kanäle kopieren und die fünften Eingangsgewichte null setzen. Die zusätzliche Information schränkt die darstellbaren Funktionen somit nicht ein. Ein schlechteres trainiertes Ergebnis ist trotzdem möglich, weil endliches DQN-Training keine Suche nach dem global besten Prädiktor garantiert.

Auf der 11×13-Karte besitzt das CNN 4.976.708 Parameter, das Motion-CNN 4.977.284. Die Differenz beträgt nur **576 Parameter**, ungefähr 0,0116 %. Eine Erklärung allein durch deutlich größere Modellkapazität ist daher schwach. Die Architektur behält das Raster ohne Pooling bei; die drei 3×3-Convolutions haben ein lokales rezeptives Feld von 7×7, die anschließende dichte Schicht kombiniert globale Information.

Der zusätzliche Kanal ist eine weitere Belegungskarte, kein expliziter Geschwindigkeitsvektor. Das Netz muss aus aktuellen und vorherigen Positionen selbst passende Bewegungsmerkmale lernen. Es erhält keine Geister-IDs, versteckten Richtungen oder RNG-Zustände. Bei Zusammenfallen mehrerer Geister werden diese im Binärbild nicht getrennt markiert. Daraus folgt aber noch kein nachgewiesener relevanter Informationsfehler auf dieser Karte. Die stochastische Richtungswahl bleibt auch mit korrekter Historie unvorhersagbar im Einzelfall.

Ein konkreter, getesteter Optimierungseffekt: **Nullinitialisierung bedeutet keine eingefrorene fünfte Ebene.** Bereits der erste Backward-Pass liefert dort nichtnull Gradienten. Bei wiederholter Geisterbelegung am Reset sind die Gradienten der aktuellen und vorherigen Geistergewichte gleich. Damit sind Anfangs-Policy und gemeinsame Gradienten vor Clipping zwar gleich, die Parametrisierung des nächsten Updates aber verschieden. Die gleiche Geisterinformation kann über zwei Gewichtsbänke angepasst werden. Allgemein kann die zusätzliche Gradientenkomponente auch das globale Norm-Clipping beeinflussen.

Die neuen Tests bestätigen gleiche initiale Losses und gemeinsame Gradienten sowohl am Reset als auch bei real bewegten Geistern einschließlich nichtterminalem Bootstrap. Sie bestätigen außerdem, dass der zusätzliche Kanal lernfähig ist. Sie beweisen nicht, dass dieser Optimierungseffekt die beobachteten −13,30 pp verursacht.

Weitere plausible, unbewiesene Mechanismen sind korrelierte Eingaben, ungünstig gelernte veraltete Gefahrenmerkmale, abweichende Zustandsbesuche durch frühe Policy-Unterschiede, Replay-Verteilung und Checkpoint-Selektion auf wiederholt denselben 100 Validierungsfällen. Der vorhandene unabhängige 1000-Episoden-Audit ist deshalb wesentlich aussagekräftiger als das Auswahlmaximum. Er darf jetzt nicht zum Nachjustieren wiederverwendet werden.

Der rohe Observation-Speicher wächst bei gleicher Übergangskapazität um 25 %: für jeweils Zustand und Folgezustand bei 30.000 Einträgen von 34,32 MB auf 42,90 MB, ohne Python-Objektoverhead. Das ist weder eine kleinere Replay-Kapazität noch bereits ein gemessener Trainingsengpass.

## 6. macOS-Implementierung und Validierung

Lokal validierte Laufzeit: **Apple M5**, native **arm64**-Prozesse, macOS 27.0.1, Homebrew-Python 3.13.15, PyTorch 2.14.1, NumPy 2.5.4, Pygame 2.6.1, Gymnasium 1.4.0, Rust/Cargo 1.99.1. Der genaue Python-Paketstand liegt unter [validation/macos-2026-10-25/requirements.txt](VALIDATION.md).

Änderungen für den lokalen Betrieb:

- `scripts/setup-macos.sh`: arm64-/Python-Prüfung, optionale Homebrew-Installation fehlender Werkzeuge, lokale `.venv`, ML-Extra, Rust-Build/-Tests und pytest. Keine automatische Studie.
- `scripts/lab.sh`: Doctor, Tests, begrenzter Smoke-Test, Viewer, Rust-Parität, Simulatorbenchmark, kurzer Profiler und Studienplanung. Jeder Aufruf funktioniert unabhängig vom aktuellen Arbeitsordner.
- `pacman_lab.macos`: Laufzeitdiagnose, Vergleich initialer CPU-/Geräte-Q-Werte, endliche Gradienten und je ein Adam-Update für alle fünf Architekturen. Batch-Smoke-Test für CNN/Motion mit beiden Engines, Checkpoint-Roundtrip und höchstens 256 Übergängen pro Lauf.
- MPS-fähige Standard-Gerätewahl für Profiler, Inspector und Studien-CLI; die Studie erlaubt nun explizit `mps` und `auto`. Explizites MPS scheitert bei fehlendem Metal-Zugriff mit einer klaren Meldung.
- `caffeinate` statt Windows-Sleep-API für Vordergrund-Marathons, einschließlich Freigabe bei Fehlern; sieben PowerShell-Start-/Status-/Stopskripte entfernt.
- Begrenzte GUI-Läufe mit `--max-frames`; Pygame-Random-Viewer sowie CNN-Motion- und GRU-Viewer werden im Test mit SDL-Dummy-Display gerendert. CNN-Checkpoints verwenden `watch`; `play --mode dqn` bleibt der historische MLP-Viewer.

MPS ist nur mit Metal-Zugriff verfügbar. Im eingeschränkten Prozess meldete PyTorch `mps_available=false`, im regulären macOS-Prozess `true`. Die Hardwaretests wurden mit echtem Metal-Zugriff ausgeführt; sie ersetzen MPS nicht durch CPU-Fallback.

Zusätzlich zu den SDL-Dummy-Regressionstests wurden beide Viewer über die neuen Shell-Launcher mit echten macOS-Fenstern für je zwei Frames gestartet und automatisch beendet. Der CNN-Viewer verwendete dabei MPS und ausschließlich den neu erzeugten temporären Motion-Smoke-Checkpoint. Beide Aufrufe waren erfolgreich; siehe [gui.log](VALIDATION.md).

### Testergebnisse

Die vollständige Einrichtung wurde über das echte Setupskript durchlaufen. Abschließend: **98 pytest-Tests bestanden, keine übersprungenen oder erwarteten Fehler**; Rust: **2 Tests bestanden**. Die erzeugte `libpacman_native.dylib` ist eine Mach-O-arm64-Bibliothek. Die abschließende pytest-Ausgabe steht in [validation/macos-2026-10-25/pytest-final.log](VALIDATION.md).

Die neue Paritätsprüfung über alle sieben Karten, 0/1/2 Geister, Bewegungstakte 1/2 und 30 Seeds pro Einstellung ergab **501.488 identische Schritte** einschließlich Beobachtungen, Rewards, Episodenenden und Diagnosen. Siehe [parity.log](VALIDATION.md).

CPU und MPS bestanden jeweils fünf Gradiententests und vier echte Batch-Läufe: zwei Architekturen × zwei Engines, **64 Übergänge und sieben Adam-Updates pro Lauf**, anschließend Evaluation und Checkpoint-Roundtrip. Neue Smoke-Dateien liegen ausschließlich in eigenen temporären Verzeichnissen; die kleinen Metadatenberichte liegen unter [smoke-cpu.json](VALIDATION.md) und [smoke-mps.json](VALIDATION.md). Ihre Anfangs-Q-Vergleiche verwenden `rtol=1e-3`, `atol=1e-4`. Das ist keine Garantie bitgleicher Lernverläufe.

Die wesentlichen neuen Tests stehen in `tests/test_v094_motion_diagnostics.py`, `tests/test_v094_study_diagnostics.py` und `tests/test_macos.py`. Die ursprüngliche Suite hatte vor den Änderungen 55 bestandene Tests und einen übersprungenen Rust-Test. Die zunächst als strikte erwartete Fehler dokumentierten Studien-Bugs sind jetzt normale bestandene Regressionstests.

### Performance: getrennte Messungen

Simulatorbenchmark, `large`, 16 Welten und 160.000 Schritte: Python **128.564 Schritte/s**, Rust **292.711 Schritte/s**, Verhältnis **2,28×**; gleiche Aktionsfolge und gleicher Pellet-Checksum. Siehe [simulation-benchmark.log](VALIDATION.md). Das ist ein kurzer Simulatorlauf und kein Nachweis für einen gleich großen Trainingsvorteil.

Zusätzlich wurden acht Profile mit je 64 Schritten und sieben Adam-Updates nacheinander ohne gleichzeitig laufende Projekt-Tests oder andere Projekt-Benchmarks ausgeführt. Architektur und Gerät sind ausdrücklich getrennt:

| Architektur | CPU/Python | CPU/Rust | MPS/Python | MPS/Rust |
|---|---:|---:|---:|---:|
| CNN | 49,07 | 146,28 | 41,15 | 313,29 |
| Motion-CNN | 146,98 | 130,96 | 307,28 | 136,04 |

Einheit: profilierte Umgebungsschritte/s im **Einzelwelt-Pfad**. Initialisierung, erste Backend-Aufrufe und Synchronisierung sind enthalten. Die starken Unterschiede bereits innerhalb ähnlicher Bedingungen zeigen, dass diese kurzen, nicht ausbalancierten Messungen keine belastbare Rangliste liefern. Insbesondere ist der daraus berechnete 7,61×-Quotient für eine Bedingung kein belegter dauerhafter Trainings-Speedup. Die Messungen dienen der Funktions- und Instrumentierungsprüfung. Für Performanceentscheidungen wären Aufwärmphase, längeres Messfenster, Wiederholungen und wechselnde Reihenfolge nötig, weiterhin ohne Modellqualitätsaussage.

Details: [PROFILE_REPORT.md](VALIDATION.md) und [pipeline-summary.json](VALIDATION.md). Die GPU-Abschnitte werden synchronisiert; CUDA/Rust-Altwerte sind nicht mit diesen MPS-Messungen gleichzusetzen.

## 7. Nächste sinnvolle ML-Experimente — nur vorgeschlagen

1. **Zuerst Originalgewichte und Kurven diagnostizieren, wenn sie exportiert vorliegen.** Hashes sichern; `best.pt` und `last.pt` vergleichen; Verlauf von Validation, TD-Loss, Q-Skala, Gradienten und Auswahlschritt prüfen. Gemeinsame neue Diagnose-Seeds verwenden. Geisterdichte, Episodenlänge, Todesarten und Pelletfortschritt auswerten.
2. **Offline-Sensitivität auf einer gemeinsamen Zustandssammlung.** Echte Historie, null gesetzte Historie und dupliziertes aktuelles Geisterbild vergleichen; Q-Änderung und Aktionswechsel messen. Auch Kanalgewichte und Aktivierungen betrachten. Solche künstlichen Eingriffe können außerhalb der Trainingsverteilung liegen; eine unmittelbare Verbesserung oder Verschlechterung beweist allein keine Kausalität.
3. **Kleine vorab festgelegte Kontrollstudie statt größerem Netz.** Unter gleichen Lernparametern und Anfangsgewichten: ursprüngliches Vierkanal-CNN; Fünfkanal-CNN mit konstant null fünftem Kanal; Fünfkanal-CNN mit dupliziertem aktuellem Geisterkanal; ursprüngliche echte Historie. Null-Kontrolle prüft den zusätzlichen Rechenpfad, Duplikat-Kontrolle trennt zusätzliche korrelierte Parametrisierung von echter Zeitinformation. Zuerst nur Funktions- und Gradiententests; ein Trainingsbudget erst separat festlegen.
4. **Erst danach eine gezielte Repräsentationsablation.** Beispielsweise aktuelle Belegung plus signierte Differenz `G_t - G_(t-1)` statt zweier Belegungen. Das wäre eine ausdrücklich neue Observation-/Experimentvariante, keine stille Fehlerkorrektur. Die aktuelle Implementierung bleibt unverändert.
5. **Auswertung vorab trennen.** Mehrere Trainingsseeds, identische Budgets, unveränderte Regeln/Rewards und definierte Checkpoint-Selektion. Ein neues finales Testfenster mit mindestens 1000 gemeinsamen Episodenseeds pro Modell nur einmal am Ende verwenden; Siegquote und Pelletquote samt Seed-Varianz berichten. 620000–620999 bleibt das bereits verwendete historische Testfenster. Die Legacy-Marathon-Standardberichte mit weniger Episoden erfüllen diese Anforderung allein nicht.

Keines dieser neuen Lernexperimente wurde gestartet. Die bisherige Evidenz rechtfertigt weder „Motion ist grundsätzlich nutzlos“ noch „es muss einen Implementierungsfehler geben“.

## 8. Unveränderte Originale

SHA-256 vor und nach der Arbeit:

```text
26e54ac6bc53d8283617a711a574345d2916ac4c38bacdae737491cfe261548f  docs/EXPERIMENT_RESULTS.md
3268478d3d98f16a14845446e1cc85cd6322f462b2822a4617cbee11d8b64a62  docs/summary_620000.json
```

Die ursprünglichen Checkpoints waren nicht Bestandteil dieses Quellbaums. Kein Test griff auf solche Gewichte zu. Die Untersuchung begann mit einem Quellbaum ohne Git-Historie. Für die öffentliche Veröffentlichung wurde anschließend ein neues Repository angelegt. Lokale Rohlogs und generierte Messdateien sind ausgeschlossen; [VALIDATION.md](VALIDATION.md) enthält die bereinigte Zusammenfassung.
