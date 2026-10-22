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
    parent = [None if o['support'] is None else ids[o['support']]
              for o in objects]
    anchor = n
    tol = 1e-8
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    pair_count = max(1, len(pairs))

    # Check the support hierarchy without fixing any search coordinates.
    order = []
    pending = set(range(n))
    done = set()
    while pending:
        ready = [i for i in sorted(pending)
                 if parent[i] is None or parent[i] in done]
        if not ready:
            raise ValueError('cyclic or missing support')
        order.extend(ready)
        done.update(ready)
        pending.difference_update(ready)

    # Edge (axis, u, v, c, owner): position[v] >= position[u] + c.
    # Room edges are mandatory. Each support owns its complete geometry.
    common = []
    for k in range(3):
        for i in range(n):
            h = size[i][k] / 2
            common.extend(((k, anchor, i, h, None),
                           (k, i, anchor, h - room[k], None)))
    for i, p in enumerate(parent):
        owner = ('s', i)
        if p is None:
            h = size[i][2] / 2
            common.extend(((2, anchor, i, h, owner),
                           (2, i, anchor, -h, owner)))
        else:
            for k in range(2):
                allowance = (size[p][k] - size[i][k]) / 2
                common.extend(((k, p, i, -allowance, owner),
                               (k, i, p, -allowance, owner)))
            height = (size[p][2] + size[i][2]) / 2
            common.extend(((2, p, i, height, owner),
                           (2, i, p, -height, owner)))

    parsed = []
    tagged = []
    for r, rel in enumerate(scene.get('relations', [])):
        kind = rel['type']
        a, b = ids[rel['a']], ids[rel['b']]
        owner = ('r', r)
        if kind == 'left_of':
            value = (size[a][0] + size[b][0]) / 2 + float(rel.get('gap', 0))
            tagged.append((0, a, b, value, owner))
        elif kind == 'near':
            value = float(rel['max_distance'])
            if value < 0:
                tagged.append((0, a, a, -value, owner))
            else:
                for k in range(2):
                    tagged.extend(((k, a, b, -value, owner),
                                   (k, b, a, -value, owner)))
        else:
            raise ValueError('unsupported relation type')
        parsed.append((kind, a, b, value))

    def relax(edges, axis):
        selected = [e for e in edges if e[0] == axis]
        d = [0.0] * (n + 1)
        pred = [None] * (n + 1)
        last = None
        for _ in range(n + 1):
            last = None
            for edge in selected:
                _, a, b, gap, _ = edge
                value = d[a] + gap
                if value > d[b] + 1e-11:
                    d[b] = value
                    pred[b] = edge
                    last = b
            if last is None:
                return [d[i] - d[anchor] for i in range(n)], None
        v = last
        fallback = {e[4] for e in selected if e[4] is not None}
        for _ in range(n + 1):
            if pred[v] is None:
                return None, fallback
            v = pred[v][1]
        start = v
        owners = set()
        for _ in range(n + 2):
            edge = pred[v]
            if edge is None:
                return None, fallback
            if edge[4] is not None:
                owners.add(edge[4])
            v = edge[1]
            if v == start:
                return None, owners
        return None, fallback

    def contained(pos):
        result = []
        for k in range(3):
            axis = []
            for i in range(n):
                low, high = size[i][k] / 2, room[k] - size[i][k] / 2
                axis.append(max(low, min(high, pos[k][i])) if low <= high
                            else room[k] / 2)
            result.append(axis)
        return result

    def assess(pos):
        supported = 0
        for i, p in enumerate(parent):
            if p is None:
                supported += abs(pos[2][i] - size[i][2] / 2) <= tol
            else:
                supported += (
                    abs(pos[2][i] - pos[2][p] -
                        (size[p][2] + size[i][2]) / 2) <= tol
                    and all(abs(pos[k][i] - pos[k][p]) <=
                            (size[p][k] - size[i][k]) / 2 + tol
                            for k in range(2)))
        collisions = []
        for a, b in pairs:
            penetration = [(size[a][k] + size[b][k]) / 2 -
                           abs(pos[k][a] - pos[k][b]) for k in range(3)]
            if min(penetration) > tol:
                collisions.append((penetration[0] * penetration[1] *
                                   penetration[2], a, b))
        bad = []
        deficit = 0.0
        for r, (kind, a, b, value) in enumerate(parsed):
            if kind == 'left_of':
                violation = value - pos[0][b] + pos[0][a]
            else:
                violation = math.hypot(pos[0][a] - pos[0][b],
                                       pos[1][a] - pos[1][b]) - value
            deficit += max(0.0, violation)
            if violation > tol:
                bad.append((violation, r))
        fraction = 1 - len(bad) / len(parsed) if parsed else 1.0
        feasibility = (0.55 * supported / n +
                       0.45 * (1 - len(collisions) / pair_count))
        key = (feasibility * (0.6 + 0.4 * fraction),
               -deficit, -sum(c[0] for c in collisions))
        return key, collisions, bad

    def cost(owner):
        if owner[0] == 's':
            return 0.55 / n
        if owner[0] == 'r':
            return 0.4 / max(1, len(parsed))
        return 0.45 / pair_count

    fallback = [[room[k] / 2] * n for k in range(3)]
    for i in order:
        p = parent[i]
        fallback[2][i] = (size[i][2] / 2 if p is None else
                          fallback[2][p] + (size[p][2] + size[i][2]) / 2)
    best = contained(fallback)
    best_key, _, _ = assess(best)
    heap = []
    seen = set()
    serial = 0

    def enqueue(disabled, extra, priority):
        nonlocal serial
        if len(seen) >= 4000:
            return
        extra = tuple(dict.fromkeys(e for e in extra if e[4] not in disabled))
        signature = (disabled, frozenset(extra))
        if signature in seen:
            return
        seen.add(signature)
        serial += 1
        burden = sum(cost(o) for o in sorted(disabled))
        heapq.heappush(heap, (-priority, burden, len(extra), serial,
                              disabled, extra))

    enqueue(frozenset(), (), 1.0)
    expanded = 0
    while heap and expanded < 800 and best_key[0] < 1 - 1e-12:
        neg_priority, _, _, _, disabled, extra = heapq.heappop(heap)
        expanded += 1
        edges = [e for e in common + tagged if e[4] not in disabled]
        edges.extend(extra)
        pos = []
        conflict = None
        for k in range(3):
            axis, conflict = relax(edges, k)
            if axis is None:
                break
            pos.append(axis)
        if conflict is not None:
            for owner in sorted(conflict, key=lambda o: (cost(o), o)):
                enqueue(disabled | {owner}, extra,
                        -neg_priority - cost(owner))
            continue
        pos = contained(pos)
        key, collisions, bad = assess(pos)
        if key > best_key:
            best_key = key
            best = [axis[:] for axis in pos]
        if best_key[0] >= 1 - 1e-12:
            break

        unresolved = [c for c in collisions
                      if ('c', c[1], c[2]) not in disabled]
        if unresolved:
            _, a, b = max(unresolved)
            owner = ('c', a, b)
            choices = []
            for k in range(3):
                distance = (size[a][k] + size[b][k]) / 2
                for u, v in ((a, b), (b, a)):
                    edge = (k, u, v, distance, owner)
                    movement = distance - pos[k][v] + pos[k][u]
                    choices.append((movement, k, u, v, edge))
            for _, _, _, _, edge in sorted(choices):
                enqueue(disabled, extra + (edge,), key[0])
            enqueue(disabled | {owner}, extra, key[0] - cost(owner))

        near_bad = [(v, r) for v, r in bad
                    if parsed[r][0] == 'near' and ('r', r) not in disabled]
        if near_bad:
            _, r = max(near_bad)
            _, a, b, radius = parsed[r]
            owner = ('r', r)
            if radius >= 0:
                dx = abs(pos[0][a] - pos[0][b])
                dy = abs(pos[1][a] - pos[1][b])
                angles = [k * math.pi / 12 for k in range(7)]
                angles.append(math.atan2(dy, dx))
                if radius > 0:
                    angles.extend((math.acos(min(1.0, dx / radius)),
                                   math.asin(min(1.0, dy / radius))))
                for angle in angles:
                    limits = (radius * math.cos(angle), radius * math.sin(angle))
                    additions = tuple((k, u, v, -limits[k], owner)
                                      for k in range(2)
                                      for u, v in ((a, b), (b, a)))
                    enqueue(disabled, extra + additions, key[0])
            enqueue(disabled | {owner}, extra, key[0] - cost(owner))

    return {o['id']: [float(best[k][i]) for k in range(3)]
            for i, o in enumerate(objects)}
# EVOLVE-BLOCK-END
