# Contact-event neighborhood repair after joint constraint search

## Complete supplied history
I read all eleven supplied proposal files in full and all fifteen matching source/score pairs. These comprise the top-level baseline, three historical baselines, their four initialization copies, four iter000 attempts, two iter001 attempts, and the current sibling attempt_00. No error logs were supplied.

All eight baseline/init snapshots use identifier-grid placement with parent-centered children. Each reports combined score 0.6738452380952381, strict success 0, support 1.0, collision freedom 0.9583333333333334, and relation satisfaction 0.21666666666666667. Their source ignores relations and gives siblings identical centers. These are structural weaknesses, not failed implementations of the later mechanisms.

Every evolved snapshot reports combined score and strict success 1.0 on the six supplied development scenes. Iter000/explore_b0/00 uses difference constraints, collision branching, and proximity rectangles (approximately 0.178 seconds); /01 adds disk projection (0.202 seconds). Iter000/explore_b1/00 uses centered graph solutions, projection, and greedy relation subsets (0.201 seconds); /01 adds circular bound propagation (0.206 seconds). Iter001/explore_b0/00 adds conflict-directed relation retraction (0.216 seconds); /01 adds support-footprint and collision waivers (0.198 seconds). The current attempt_00 adds joint XYZ search and retractable complete support contacts (0.202 seconds). These are analytic AABB observations, with Blender verification false. There is no observed failed evolved attempt motivating a bug retry.

## New mechanism
Retain joint XYZ conflict search as a first-stage constructor. If its incumbent remains imperfect, switch representations: search directly over rigid translations of individual objects and complete support subtrees. Along one axis, enumerate geometric events where the actual discrete objective can change: room boundaries, collision contact faces, support-footprint boundaries, exact support heights, left-of thresholds, and roots of near-circle intersections with the translation line. Evaluate those event translations against every actual obligation, including obligations previously waived by graph search.

A subtree move preserves relative positions and internal contacts while moving a coordinated assembly. Singleton moves permit repair of an individual contact. A move can simultaneously recover several waived obligations without reconstructing their graph decisions. The repair stage retains only improvements under coordinate-based scoring and uses fixed sweep and candidate budgets. It keeps the graph incumbent if no better candidate is found.

This is a new combination of successful joint constraint search with geometric neighborhood optimization. Previous attempts add constraints, project onto enforced constraints, or monotonically retract obligations. None enumerates direct rigid-subtree translation events against the full objective after graph search. This deliberately explores a different mechanism rather than another proximity-angle or retraction-cost adjustment.

## Expected benefit and risk
The new stage may recover support or relations incidentally lost during bounded graph search, and may move an entire assembly past a local packing obstruction. Supplied measurements establish no improvement for this mechanism. Coordinate-wise event moves can miss coordinated multi-axis rearrangements and moves requiring temporary score degradation. Subtree neighborhoods can be expensive, so candidate evaluations are capped. Floating-point tangencies remain a risk; candidates are clamped to room bounds and assessed independently. Oversized objects cannot be made room-contained without changing immutable dimensions.

Only the standard library is used. No execution, compilation, evaluation, or performance improvement is claimed.
