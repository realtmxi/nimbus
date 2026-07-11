# Agent / Developer Notes

GPU experiments (real vLLM/SGLang runs, knee sweeps, bistability) require a GPU
host. Configure the host out-of-band (e.g. an SSH alias in your local
`~/.ssh/config`); do not commit machine names, usernames, or absolute scratch
paths into the repo.

No-GPU work (mock mode, unit tests, offline analysis) runs anywhere — see the
README "Validate Core Logic" section.

## GPU hygiene (shared boxes)

- **Kill every vLLM/SGLang server you started as soon as you are done with it.**
  The GPU boxes are shared; a forgotten server pins ~90GB of VRAM and blocks
  teammates. Do this at the end of every experiment session, not "later".
- Kill by PID (record the PID at launch). Do NOT `pkill -f "vllm serve ..."`
  over ssh with a pattern that also appears in your own ssh command line — it
  kills your own shell. Note `vllm serve` spawns children (APIServer,
  EngineCore); after killing the parent, verify with
  `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`
  that the memory is actually released, and kill surviving children by PID.
- Before grabbing a GPU, check `nvidia-smi` for other users' processes and pick
  a free device; never kill processes that are not yours.

## GPU device preference on the 3-GPU experiment node (Murphy, 2026-07-10)

- **Default to the THIRD GPU (`CUDA_VISIBLE_DEVICES=2`) for experiments.**
  GPU0/GPU1 have shown reliability problems; GPU0 in particular wedged twice
  in July 2026 (`cudaErrorLaunchFailure` under load, then NVML handle lost
  again while idle — "Unable to determine the device handle for GPU0"). A
  wedged device poisons driver-level CUDA init for NEW processes on ALL GPUs
  (vLLM startup also enumerates every physical device), so avoiding the flaky
  devices avoids whole-box reboots.
- Observed record for calibration: GPU0 failed twice; GPU1 ran all router
  experiments without a device-level fault (its only failures were collateral
  from GPU0's wedge). Treat GPU1 as second choice, GPU0 as last resort until
  the hardware diagnosis (kernel Xid codes) says otherwise.
- GPU2 is also the default in teammates' serving scripts — check
  `nvidia-smi` for their processes first and coordinate before taking it.
