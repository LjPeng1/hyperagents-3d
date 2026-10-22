# Third-party components

The unmodified HyperAgents source and this derivative research execution variant
retain the repository's [CC BY-NC-SA 4.0 license](LICENSE.md). This is research
code with non-commercial and share-alike conditions, not unrestricted open source.

The six exact-byte files under `portable/task/` are copied from
[LjPeng1/dream-rsi-3d](https://github.com/LjPeng1/dream-rsi-3d), commit
`361706351ccfabfca9b1829c9ce44177923271f5`, directory `tasks/blender_layout/`.
Those files retain their original Apache-2.0 licensing. The full license is
provided in `portable/task/LICENSE-APACHE-2.0.txt`; it is not a benchmark input.

The portable runner imports the installed `dream_rsi` package and its Codex
transport instead of copying it. DREAM-RSI is Copyright 2026 Google LLC,
Apache-2.0. Our comparison checkout includes the public Windows and structured
transport additions at the commit above. The upstream Google Research revision
is `8ec35bd2538220517ce1bfca4282db21b8c518db`.

The HyperAgents upstream revision audited for this fork is
`59a68f672dfb92c74aeb7e61535d776fb36e172d`.
