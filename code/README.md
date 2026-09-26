# code/

This is a frozen snapshot of the agent implementation as of the `reflexion_V3` result (76.39% Easy, 53.44% Hard, 57.11% overall on DABstep). It is copied here for the write-up, not maintained here.

It will not run standalone from a fresh clone of this repo. It needs the DABstep dataset, which is not committed here and a `.env` with credentials for an OpenAI or Azure OpenAI-compatible endpoint (`ENDPOINT`, `SUBSCRIPTION_KEY`, and related variables read via `os.environ`).

For setup and how to run a task or the full set, see the top-level `README.md` in this repository.