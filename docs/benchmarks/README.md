# How the files here were made

The tables and their reading are in `docs/RESULTS.md`; this page only says which command
wrote which file. Files not listed here name their command in their own header
(`gpu_benchmark_all_*.md`) or in the `RESULTS.md` section that cites them.

## `bench_*`, `session_*`, `conditioning_*` (`tools/benchmark.py`)

All were run on rented GPU machines through `tools/gpu_job.sh` with `GPUSONAR_JOB=benchmark`
(`docs/HANDOFF.md`), which runs `tools/benchmark.py --backend cuda` with `--scale`
`GPUSONAR_SCALE` (default 2), `--pixel` `GPUSONAR_PIXEL` (default 0.05 m) and `--modes`
`GPUSONAR_MODES` (default `sas,fls,flank`). The job writes no file: the tool prints its
result as one line `BENCHMARK_JSON {...}` at the end of the log, and that JSON was saved
under the name below. The machine is in each file (`devices`); every file here has
`scale: 2.0`.

| Files | Command (inside the job) | Tool as of | What changed between rounds |
| --- | --- | --- | --- |
| `before_pretest/bench_{t4,l4,a10g,a100}.json` | `tools/benchmark.py --backend cuda --scale 2` (modes `sas,fls,flank`, 5 cm pixels) | `b9d613f` (2026-11-16), committed there as `bench_*.json` and moved here in `2dfd853` | the kernel before the float32 pre-test |
| `bench_{t4,l4,a10g,a100,a10gx2}.json` | the same | `2dfd853` (2026-11-16) | after the pre-test. `a10gx2`: a machine with two A10G; with more than one device the tool adds the `multi_gpu_*` entries itself |
| `session_{t4,l4,a100,a10gx2}.json` | the same | `c9d4067` (T4, two A10G) and `33171a0` (L4, A100), 2026-11-16 | the tool also times `BackprojectSession` (`session_*` entries) |
| `session_{t4,l4,a100,a10gx2}_2cm.json` | `tools/benchmark.py --backend cuda --scale 2 --pixel 0.02 --modes sas` (`GPUSONAR_PIXEL=0.02 GPUSONAR_MODES=sas`) | `c9d4067` (two A10G), `33171a0` (the others) | the SAS image at 2 cm (6400 × 3000) |
| `conditioning_{t4,a10g}.json` | `tools/benchmark.py --backend cuda --scale 2 --modes conditioning` (`GPUSONAR_MODES=conditioning`) | `001de53` (2026-11-16) | baseband conversion and pulse compression, host against device |

"Tool as of" is the commit that added the files; rerunning with a later tree times later
code (the kernel has changed since), so a rerun is a new measurement, not a regeneration.
