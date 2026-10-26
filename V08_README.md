# v0.8 – 16-world batched CNN/GRU training

**Target:** Windows 10, i7-4770, GTX 1080 Ti (PyTorch 2.8.0+cu126), 32 GB RAM. Extends v0.7 without modifying its existing checkpoints or game rules.

## What changed

- `pacman_lab/cnn_batch.py`: 16 independent worlds and a single batched PyTorch policy inference per rollout tick, using `NativeBatch` when `--engine rust` is selected.
- Four architectures unchanged: large MLP, CNN, CNN+four-frame history, CNN+GRU; all Double DQN. The recurrent agent stores separate GRU hidden states and separate episode buffers for every world. Recurrent experience replay never combines episodes or worlds.
- Batched evaluation runs deterministic held-out seeds and produces the same values as v0.7 sequential evaluation on reference tests. We compare **environment transitions**, not number of batch calls.
- `--worlds 1` selects the old unbatched trainer. `--worlds 16` selects the new trainer; `Start-CNN-Marathon.ps1` defaults to 16. **This is batched inference, not Rust multithreading**: the Rust backend processes the 16 simulations in a single FFI call but currently loops over worlds on the CPU. Since simulation was ~2% of profiled training time, the GPU bottleneck has priority.
- Independent suite paths and resumable new-version checkpoints. Like v0.7, restart recovers optimizer/model/target and step count, **but not replay, RNG or in-progress episodes**. This is a warm restart, not deterministic exact resumption.
- New `batch_bench` measures **complete training** with real optimizer updates, including validation and saving. This must be run on the target GPU to establish actual benefit; the CPU packaging environment cannot predict a CUDA speedup.

## Safe upgrade from a running v0.7 marathon

**Do not overwrite source files while the old marathon is running.** Your existing `runs/` directory and `.venv/` are deliberately absent from this ZIP. To stop the old marathon, before extracting the upgrade run in a PowerShell window (make sure there are no other Pac-Man RL jobs you want to keep):

```powershell
cd C:\path\to\pacman_rl_lab
Get-CimInstance Win32_Process | Where-Object {
  $_.Name -match '^python(w)?\.exe$' -and
  $_.CommandLine -match 'pacman_lab\.cnn_(marathon|train)'
} | Select-Object ProcessId,CommandLine
```

When ready, stop **only** the two processes matching this project's Python virtual environment and the old marathon modules:

```powershell
$python = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$jobs = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.CommandLine -like "*$python*" -and
    $_.CommandLine -match 'pacman_lab\.cnn_(marathon|train)'
})
$jobs | Select-Object ProcessId,CommandLine | Format-List
# Check the displayed commands before executing this line:
$jobs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

This interrupts the previous run, which is still preserved under `runs/cnn_.../`. The trainer uses atomic checkpoint replacement. It is preferable to stop just after a logged checkpoint rather than in the middle of a save. **Do not delete the old run**. After upgrading, `Stop-CNN-Marathon.ps1` provides a dry run first and `-Stop` to stop this project's workers.

## Install and validate

Extract the new ZIP into `C:\path\to\` (one top-level `pacman_rl_lab` folder), overwriting source files without deleting `.venv`, `runs`, or `native/pacman_native/target`.

```powershell
cd C:\path\to\pacman_rl_lab
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pacman_lab.native_bench verify --episodes 5
```

The Rust engine code has not changed, so you should not need to recompile the existing `target-cpu=native` DLL. But **the native DLL must be present**. `native_bench verify` exercises exact random gameplay parity.

## Benchmark full training before restarting

```powershell
.\.venv\Scripts\python.exe -m pacman_lab.batch_bench --engine rust --device cuda --architecture cnn_history4 --worlds 16 --steps 3000
.\.venv\Scripts\python.exe -m pacman_lab.batch_bench --engine rust --device cuda --architecture cnn_gru --worlds 16 --steps 3000
```

Each command creates a new `runs/batch_benchmark_.../throughput.json` and prints the end-to-end speed ratio. Short-run initialization, warm-up, held-out evaluation and checkpoint I/O bias results; they are useful for sanity checking but not a guarantee of improved long-run throughput.

## New marathon (only after old run is stopped)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-CNN-Marathon.ps1 -Engine rust -Worlds 16 -Steps 1000000 -Repeats 3
powershell -NoProfile -ExecutionPolicy Bypass -File .\Status-CNN-Marathon.ps1
```

Runs: four network architectures × three seeds × 1M **environment steps** = 12M training transitions. The new suite will be saved under `runs/cnn_<timestamp>`, so previous results are not overwritten. GPU/CPU learning progress, model files and best/last checkpoints saved automatically. Upload `UPLOAD_THIS_CNN_report.zip` after it finishes. Leave Windows awake/plugged in; automatic sleep prevention works only while marathon runner is alive.

### Lower-memory alternative / correctness fallback

Use `-Worlds 4` or `-Worlds 8` if 16 lowers speed on your GTX 1080 Ti. Test both; bigger batches are not automatically faster. `-Worlds 1` uses the original unbatched code for troubleshooting. The older `Start-Pipeline-Profile.ps1` still profiles the **v0.7 single-world trainer**; use `batch_bench` for v0.8 throughput.

### Limitations

- Cannot claim higher training throughput until measured on your Windows/CUDA setup. CPU-only smoke tests do not represent GTX 1080 Ti speed.
- Parallel experiences change the order and diversity of examples; even with the same seed, trained parameters need not match v0.7 bit-for-bit. Use held-out results and repeated seeds to compare learning, not only speed.
- An episode may be partially advanced on the final incomplete batch of a nondivisible budget; only the requested number of transitions are stored and counted.
- Across architectures, capacity is not matched. MLP vs CNN improvement cannot be ascribed to inductive bias alone.
