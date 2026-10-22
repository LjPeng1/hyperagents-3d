# Constraint-graph search with geometric branching

## Supplied history
I reviewed the complete supplied baseline proposal, both baseline source copies, and both score files. The initialization and baseline contain identical code and measured results: combined score 0.6738452380952381, strict success 0, support 1.0, collision-free fraction 0.9583333333333334, and relation satisfaction 0.21666666666666667. No other historical proposals or failure logs were supplied.

The baseline's source explains its shortcomings: base placement depends on sorted identifiers rather than relations, and every supported object is centered on its parent, allowing siblings to overlap. These are structural limitations rather than an execution failure.

## Proposed mechanism
Replace the identifier grid with bounded difference-constraint graphs for both horizontal axes. Room containment, parent-footprint containment, and left-of relations become graph inequalities. Heights are derived exactly from the support hierarchy. Search resolves each remaining collision by branching over the four possible horizontal separating directions. Violated near relations generate alternative rectangular distance bounds lying inside the required circle. Graph infeasibility prunes branches before further expansion. Candidate selection uses the documented objective calculated from the supplied scene geometry, without accessing evaluation state.

This combines explicit relation reasoning with disjunctive packing; neither mechanism appears in the supplied baseline. A bounded deterministic search and a valid-coordinate fallback keep the solver reusable and limit computation.

## Expected benefit and risk
Explicit left-of constraints should improve relation satisfaction, while footprint-aware collision branching should address overlapping siblings without sacrificing support on feasible instances. Near constraints can rearrange both supported objects and their parents through the same graph. Risks include search truncation, discrete near-bound choices, and greedy selection when all requested relations cannot coexist. Impossible support geometry receives only a fallback layout. The implementation has not been executed or evaluated, and no improvement is claimed as measured.