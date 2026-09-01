# Claude / agent instructions for this repo

- **GPU hygiene:** any vLLM/SGLang server started for an experiment MUST be
  killed as soon as it is no longer needed (shared GPU boxes — free the VRAM
  for teammates). Record the launch PID, kill by PID when done, then verify
  with `nvidia-smi` that the memory is released (vllm spawns APIServer /
  EngineCore children that can survive the parent). Never `pkill -f` with a
  pattern contained in your own ssh command line, and never touch other
  users' processes.
- Do not commit machine names, usernames, SSH aliases, absolute `/scratch`
  paths, or API keys. Keys live in the gitignored `.env`.
- See `AGENTS.md` for the full shared-box conventions.
