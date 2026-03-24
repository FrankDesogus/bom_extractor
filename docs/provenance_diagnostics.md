# Provenance diagnostics notes

Final-row provenance is collected as non-invasive metadata under `row.metadata["diagnostic_provenance"]`.

Collection points:
- `weak_map_columns` (`weak_map_columns` stage).
- lane reconstruction and lane assignment (`anchor_reconstruction`, `lane_assignment` stages).
- row-boundary merge events (`row_boundary_engine` stage merge flags + boundary-adjusted confidence snapshot).
- final multiline stitch merge events (`final_stitch` stage field writes + merge flags).

A debug-only artifact `rows.provenance.jsonl` is emitted with row id and the provenance payload.
