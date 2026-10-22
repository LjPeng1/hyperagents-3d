# Joint XYZ search with retractable support contacts

## Complete supplied history
I reviewed all ten supplied proposal files in full, together with all fourteen source/score pairs. These include the top-level baseline, the three historical baseline directories and their initialization copies, four iter000 attempts, and two iter001 attempts. No error logs were supplied.

All eight baseline/init snapshots use the same identifier grid and parent-centered children. Their measured combined score is 0.6738452380952381, strict success is 0, support is 1.0, collision freedom is 0.9583333333333334, and relation satisfaction is 0.21666666666666667. Their source ignores relations and gives siblings identical centers: structural limitations, not execution failures.

The six evolved attempts all report combined score and strict success of 1.0 on the six supplied development scenes. Iter000/b0/00 uses difference graphs, collision branching, and proximity rectangles; b0/01 adds disk projection. Iter000/b1/00 uses centered graph solutions, projection, and greedy relation subsets; b1/01 adds circular bound propagation. Iter001/b0/00 introduces conflict-directed relation retraction; b0/01 extends retraction to support footprints and collision obligations. Their reported evaluation times are approximately 0.178, 0.202, 0.201, 0.206, 0.216, and 0.198 seconds. All are analytic AABB evaluations with Blender verification false. No evolved attempt has an observed failure requiring a bug retry.

## New mechanism
The history clusters around horizontal graph search with unchanged observed outcomes. Every evolved implementation computes heights before search; even the soft-geometry implementation can release only support footprints, leaving vertical positions fixed.

This proposal searches all three coordinates jointly. A support obligation owns its XY containment constraints and both inequalities defining its exact vertical contact. Floor contact is represented similarly for roots. Initially every support obligation is enforced. Positive-cycle conflicts can retract an entire support obligation, allowing its object and still-supported descendants to move vertically. Collision resolution branches over all six separating directions, plus the existing explicit overlap waiver. Thus a difficult horizontal packing can be replaced by a different vertical arrangement, with any lost support counted honestly by coordinate-based scoring. Support assignments themselves remain unchanged.

This combines successful conflict-directed soft-constraint search with a previously untried vertical degree of freedom. It is not another proximity refinement or a repeat of footprint-only relaxation. Room containment remains mandatory on all axes; all actual supports, collisions, and relations are assessed independently of enforcement state. Proximity retains the earlier rectangular branches.

## Expected benefit and risk
Joint XYZ search may find better partial layouts when accepting one broken support contact allows several collisions or relation conflicts to be resolved. Fully feasible scenes can retain every prescribed contact. The supplied measurements do not establish a benefit for this change.

Risks include more branching, heuristic priorities missing useful support tradeoffs, finite proximity-angle coverage, and numerical sensitivity. Search uses fixed expansion and state budgets and only the Python standard library. Oversized objects remain intrinsically impossible to contain. This source has not been executed or evaluated; no correctness or performance improvement is claimed.