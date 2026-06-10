# Agent / Developer Notes

GPU experiments (real vLLM/SGLang runs, knee sweeps, bistability) require a GPU
host. Configure the host out-of-band (e.g. an SSH alias in your local
`~/.ssh/config`); do not commit machine names, usernames, or absolute scratch
paths into the repo.

No-GPU work (mock mode, unit tests, offline analysis) runs anywhere — see the
README "Validate Core Logic" section.
