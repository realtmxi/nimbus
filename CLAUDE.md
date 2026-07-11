# Claude / agent instructions for this repo

- **GPU hygiene:** any vLLM/SGLang server started for an experiment MUST be
  killed as soon as it is no longer needed (shared GPU boxes — free the VRAM
  for teammates). Record the launch PID, kill by PID when done, then verify
  with `nvidia-smi` that the memory is released (vllm spawns APIServer /
  EngineCore children that can survive the parent). Never `pkill -f` with a
  pattern contained in your own ssh command line, and never touch other
  users' processes.
- **GPU choice on the 3-GPU experiment node: default to `CUDA_VISIBLE_DEVICES=2`**
  (the third GPU). GPU0/GPU1 have shown reliability problems — GPU0 wedged
  twice (July 2026) and a wedged device breaks CUDA init for new processes on
  the whole box, forcing a reboot. Order of preference: GPU2 → GPU1 → GPU0.
  GPU2 is also teammates' default serving device: check `nvidia-smi` for
  their processes before taking it.
- Do not commit machine names, usernames, SSH aliases, absolute `/scratch`
  paths, or API keys. Keys live in the gitignored `.env`.
- See `AGENTS.md` for the full shared-box conventions.
