"""Initial general heuristic. Only the marked block is evolved."""
# EVOLVE-BLOCK-START
def solve(scene):
    import math
    import heapq

    objects = scene['objects']
    n = len(objects)
    if not n:
        return {}
    ids = {o['id']: i for i, o in enumerate(objects)}
    size = [list(map(float, o['size'])) for o in objects]
    room = list(map(float, scene['room']))
    parent = [None if o['support'] is None else ids[o['support']] for o in objects]
    relations = scene.get('relations', [])
    anchor = n
    margin = 1e-7
    tol = 1e-8
    z = [None] * n
    pending = set(range(n))
    while pending:
        ready = [i for i in sorted(pending) if parent[i] is None or z[parent[i]] is not None]
        if not ready:
            raise ValueError('cyclic or missing support')
        for i in ready:
            p = parent[i]
            z[i] = size[i][2] / 2 if p is None else z[p] + (size[p][2] + size[i][2]) / 2
            pending.remove(i)
    z = [max(size[i][2] / 2, min(room[2] - size[i][2] / 2, z[i])) for i in range(n)]

    # Edge: (axis, source, destination, lower bound, relation owner).
    # Owner -1 denotes mandatory geometry or a packing decision.
    common = []
    for k in range(2):
        for i in range(n):
            h = size[i][k] / 2
            common.extend(((k, anchor, i, h, -1), (k, i, anchor, h - room[k], -1)))
            p = parent[i]
            if p is not None:
                allowance = (size[p][k] - size[i][k]) / 2
                common.extend(((k, p, i, -allowance, -1), (k, i, p, -allowance, -1)))
    tagged = []
    parsed = []
    for r, rel in enumerate(relations):
        a, b = ids[rel['a']], ids[rel['b']]
        kind = rel['type']
        if kind == 'left_of':
            value = (size[a][0] + size[b][0]) / 2 + float(rel.get('gap', 0.0))
            tagged.append((0, a, b, value, r))
        elif kind == 'near':
            value = float(rel['max_distance'])
            if value < 0:
                tagged.append((0, a, a, -value, r))
            else:
                for k in range(2):
                    tagged.extend(((k, a, b, -value, r), (k, b, a, -value, r)))
        else:
            raise ValueError('unsupported relation type')
        parsed.append((kind, a, b, value))

    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)
             if (size[a][2] + size[b][2]) / 2 - abs(z[a] - z[b]) > tol]

    def relax(edges, axis):
        selected = [e for e in edges if e[0] == axis]
        d = [0.0] * (n + 1)
        pred = [None] * (n + 1)
        last = None
        for _ in range(n + 1):
            last = None
            for edge in selected:
                _, a, b, gap, _ = edge
                if d[a] + gap > d[b] + 1e-11:
                    d[b] = d[a] + gap
                    pred[b] = edge
                    last = b
            if last is None:
                return [d[i] - d[anchor] for i in range(n)], None
        # Following predecessors enters the positive cycle responsible for failure.
        v = last
        for _ in range(n + 1):
            if pred[v] is None:
                return None, {e[4] for e in selected if e[4] >= 0}
            v = pred[v][1]
        start = v
        owners = set()
        for _ in range(n + 2):
            edge = pred[v]
            if edge is None:
                return None, {e[4] for e in selected if e[4] >= 0}
            if edge[4] >= 0:
                owners.add(edge[4])
            v = edge[1]
            if v == start:
                break
        return None, owners

    def assess(pos):
        supported = 0
        for i, p in enumerate(parent):
            if p is None:
                supported += abs(z[i] - size[i][2] / 2) <= tol
            else:
                supported += (abs(z[i] - z[p] - (size[p][2] + size[i][2]) / 2) <= tol
                              and all(abs(pos[k][i] - pos[k][p]) <= (size[p][k] - size[i][k]) / 2 + tol for k in range(2)))
        collisions = []
        for a, b in pairs:
            ox = (size[a][0] + size[b][0]) / 2 - abs(pos[0][a] - pos[0][b])
            oy = (size[a][1] + size[b][1]) / 2 - abs(pos[1][a] - pos[1][b])
            if min(ox, oy) > tol:
                collisions.append((ox * oy, a, b))
        bad = []
        deficit = 0.0
        for r, (kind, a, b, value) in enumerate(parsed):
            violation = value - pos[0][b] + pos[0][a] if kind == 'left_of' else math.hypot(pos[0][a] - pos[0][b], pos[1][a] - pos[1][b]) - value
            deficit += max(0.0, violation)
            if violation > tol:
                bad.append((violation, r))
        fraction = 1 - len(bad) / len(parsed) if parsed else 1.0
        feasibility = 0.55 * supported / n + 0.45 * (1 - len(collisions) / max(1, n * (n - 1) // 2))
        key = (feasibility * (0.6 + 0.4 * fraction), -deficit, -sum(c[0] for c in collisions))
        return key, collisions, bad

    def contained(pos):
        return [[max(size[i][k] / 2, min(room[k] - size[i][k] / 2, pos[k][i])) for i in range(n)] for k in range(2)]

    best = contained([[room[k] / 2] * n for k in range(2)])
    best_key, _, _ = assess(best)
    heap = []
    seen = set()
    serial = 0

    def enqueue(active, extra, priority=1.0):
        nonlocal serial
        if len(seen) >= 2600:
            return
        extra = tuple(sorted(set(e for e in extra if e[4] < 0 or e[4] in active)))
        signature = (active, extra)
        if signature in seen:
            return
        seen.add(signature)
        serial += 1
        heapq.heappush(heap, (-priority, len(parsed) - len(active), len(extra), serial, active, extra))

    enqueue(frozenset(range(len(parsed))), ())
    expanded = 0
    while heap and expanded < 650 and best_key[0] < 1 - 1e-12:
        _, _, _, _, active, extra = heapq.heappop(heap)
        expanded += 1
        edges = common + [e for e in tagged if e[4] in active] + list(extra)
        pos = []
        conflict = None
        for k in range(2):
            axis, conflict = relax(edges, k)
            if axis is None:
                break
            pos.append(axis)
        if conflict is not None:
            # Every feasible descendant must retract an owner on this cycle,
            # unless it changes an earlier packing choice (a separate branch).
            for r in sorted(conflict):
                enqueue(active - {r}, extra, 1.0 - 0.4 * (len(parsed) - len(active) + 1) / max(1, len(parsed)))
            continue
        pos = contained(pos)
        key, collisions, bad = assess(pos)
        if key > best_key:
            best_key = key
            best = [axis[:] for axis in pos]
        if best_key[0] >= 1 - 1e-12:
            break
        if collisions:
            _, a, b = max(collisions)
            choices = []
            for k in range(2):
                separation = (size[a][k] + size[b][k]) / 2 + margin
                for u, v in ((a, b), (b, a)):
                    choices.append((separation - pos[k][v] + pos[k][u], (k, u, v, separation, -1)))
            for _, edge in sorted(choices):
                enqueue(active, extra + (edge,), key[0])
        # Repair proximity even while collisions exist, offering a different
        # ordering of geometric commitments. Retraction is a separate choice.
        near_bad = [(v, r) for v, r in bad if r in active and parsed[r][0] == 'near']
        if near_bad:
            _, r = max(near_bad)
            _, a, b, radius = parsed[r]
            if radius >= 0:
                dx = abs(pos[0][a] - pos[0][b])
                dy = abs(pos[1][a] - pos[1][b])
                angles = [k * math.pi / 12 for k in range(7)]
                angles.append(math.atan2(dy, dx))
                if radius > 0:
                    angles.extend((math.acos(min(1.0, dx / radius)), math.asin(min(1.0, dy / radius))))
                for angle in angles:
                    limits = (radius * math.cos(angle), radius * math.sin(angle))
                    additions = tuple((k, u, v, -limits[k], r) for k in range(2) for u, v in ((a, b), (b, a)))
                    enqueue(active, extra + additions, key[0])
            enqueue(active - {r}, extra, key[0])

    return {obj['id']: [float(best[0][i]), float(best[1][i]), float(z[i])] for i, obj in enumerate(objects)}
# EVOLVE-BLOCK-END
