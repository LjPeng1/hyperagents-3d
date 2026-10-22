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
    sizes = [list(map(float, o['size'])) for o in objects]
    room = list(map(float, scene['room']))
    parents = [None if o['support'] is None else ids[o['support']]
               for o in objects]
    relations = scene.get('relations', [])
    heights = [None] * n
    remaining = set(range(n))
    while remaining:
        ready = [i for i in remaining
                 if parents[i] is None or heights[parents[i]] is not None]
        if not ready:
            raise ValueError('cyclic or missing support')
        for i in ready:
            p = parents[i]
            heights[i] = sizes[i][2] / 2 if p is None else (
                heights[p] + (sizes[p][2] + sizes[i][2]) / 2)
            remaining.remove(i)

    eps = 1e-6
    source = n
    common = [[], []]
    for axis in range(2):
        for i in range(n):
            half = sizes[i][axis] / 2
            common[axis].append((source, i, half))
            common[axis].append((i, source, half - room[axis]))
            p = parents[i]
            if p is not None:
                allowance = (sizes[p][axis] - sizes[i][axis]) / 2
                common[axis].append((p, i, -allowance))
                common[axis].append((i, p, -allowance))

    directional = []
    nearby = []
    for r in relations:
        a, b = ids[r['a']], ids[r['b']]
        if r['type'] == 'left_of':
            directional.append((a, b, (sizes[a][0] + sizes[b][0]) / 2
                                + float(r.get('gap', 0)) + eps))
        elif r['type'] == 'near':
            nearby.append((a, b, max(0.0, float(r['max_distance']) - eps)))

    vertical_pairs = []
    for i in range(n):
        for j in range(i):
            if abs(heights[i] - heights[j]) < (
                    sizes[i][2] + sizes[j][2]) / 2 - 1e-9:
                vertical_pairs.append((j, i))

    def longest(edges, reverse=False):
        values = [0.0] * (n + 1)
        for iteration in range(n + 1):
            changed = False
            for u, v, d in edges:
                if reverse:
                    u, v = v, u
                target = values[u] + d
                if target > values[v] + 1e-11:
                    values[v] = target
                    changed = True
            if not changed:
                origin = values[source]
                sign = -1 if reverse else 1
                return [sign * (v - origin) for v in values[:n]]
        return None

    def coordinates(graphs):
        axes = []
        for edges in graphs:
            low = longest(edges)
            if low is None:
                return None
            high = longest(edges, True)
            if high is None:
                return None
            axes.append([(a + b) / 2 for a, b in zip(low, high)])
        return axes

    def overlaps(pos):
        found = []
        for i, j in vertical_pairs:
            dx = (sizes[i][0] + sizes[j][0]) / 2 - abs(pos[0][i] - pos[0][j])
            dy = (sizes[i][1] + sizes[j][1]) / 2 - abs(pos[1][i] - pos[1][j])
            if dx > 1e-8 and dy > 1e-8:
                found.append((dx * dy, i, j))
        found.sort(reverse=True)
        return found

    def assess(pos):
        supported = 0
        for i, p in enumerate(parents):
            if p is None or all(
                    abs(pos[k][i] - pos[k][p]) <=
                    (sizes[p][k] - sizes[i][k]) / 2 + 1e-8
                    for k in range(2)):
                supported += 1
        collisions = overlaps(pos)
        satisfied = 0
        deficit = 0.0
        for r in relations:
            a, b = ids[r['a']], ids[r['b']]
            if r['type'] == 'left_of':
                violation = (pos[0][a] - pos[0][b]
                             + (sizes[a][0] + sizes[b][0]) / 2
                             + float(r.get('gap', 0)))
            elif r['type'] == 'near':
                violation = math.hypot(pos[0][a] - pos[0][b],
                                       pos[1][a] - pos[1][b]) - float(r['max_distance'])
            else:
                continue
            satisfied += violation <= 1e-8
            deficit += max(0.0, violation)
        fraction = satisfied / len(relations) if relations else 1.0
        pairs = max(1, n * (n - 1) // 2)
        feasibility = 0.55 * supported / n + 0.45 * (1 - len(collisions) / pairs)
        score = feasibility * (0.6 + 0.4 * fraction)
        return (score, -deficit, -sum(c[0] for c in collisions)), collisions

    def project_near(pos, graphs):
        if not nearby:
            return pos
        if all(math.hypot(pos[0][a] - pos[0][b], pos[1][a] - pos[1][b])
               <= limit for a, b, limit in nearby):
            return pos
        # Dykstra projections onto linear halfspaces and XY distance disks.
        work = [pos[0][:] + [0.0], pos[1][:] + [0.0]]
        corrections = [[0.0] * len(g) for g in graphs]
        disk_corrections = [[0.0, 0.0] for _ in nearby]
        for iteration in range(180):
            movement = 0.0
            for axis, edges in enumerate(graphs):
                v = work[axis]
                for e, (a, b, d) in enumerate(edges):
                    if a == b:
                        continue
                    old = corrections[axis][e]
                    va = v[a] - old if a != source else 0.0
                    vb = v[b] + old if b != source else 0.0
                    count = int(a != source) + int(b != source)
                    adjustment = max(0.0, d - (vb - va)) / count
                    na = va - adjustment
                    nb = vb + adjustment
                    if a != source:
                        movement = max(movement, abs(na - v[a]))
                        v[a] = na
                    if b != source:
                        movement = max(movement, abs(nb - v[b]))
                        v[b] = nb
                    corrections[axis][e] = -adjustment
            for e, (a, b, limit) in enumerate(nearby):
                if a == b:
                    continue
                oldx, oldy = disk_corrections[e]
                ax, ay = work[0][a] + oldx, work[1][a] + oldy
                bx, by = work[0][b] - oldx, work[1][b] - oldy
                dx, dy = ax - bx, ay - by
                distance = math.hypot(dx, dy)
                scale = max(0.0, 1 - limit / distance) / 2 if distance else 0.0
                cx, cy = dx * scale, dy * scale
                for axis, av, bv, correction in ((0, ax, bx, cx), (1, ay, by, cy)):
                    movement = max(movement, abs(work[axis][a] - av + correction),
                                   abs(work[axis][b] - bv - correction))
                    work[axis][a] = av - correction
                    work[axis][b] = bv + correction
                disk_corrections[e] = [cx, cy]
            if movement < 1e-10:
                break
        for axis, edges in enumerate(graphs):
            if any(work[axis][b] - work[axis][a] < d - 1e-9
                   for a, b, d in edges):
                return pos
        return [work[0][:n], work[1][:n]]

    # A finite, room-contained fallback also covers geometrically impossible input.
    fallback = [[room[k] / 2 for _ in objects] for k in range(2)]
    best_pos = fallback
    best_key, _ = assess(fallback)

    def consider(pos):
        nonlocal best_pos, best_key
        # Avoid returning minute numerical excursions beyond room boundaries.
        pos = [[min(max(sizes[i][k] / 2, pos[k][i]),
                    room[k] - sizes[i][k] / 2) for i in range(n)]
               for k in range(2)]
        key, collisions = assess(pos)
        if key > best_key:
            best_key = key
            best_pos = [pos[0][:], pos[1][:]]
        return key, collisions, pos

    # Full relation set first; alternate maximal feasible subsets provide recovery
    # when contradictory or overlong orderings cannot all fit in the room.
    seeds = []
    seen_seeds = set()
    orders = [list(range(len(directional))),
              list(reversed(range(len(directional)))),
              sorted(range(len(directional)), key=lambda i: directional[i][2])]
    full = (common[0] + directional, common[1][:])
    if coordinates(full) is not None:
        seeds.append(full)
        seen_seeds.add(tuple(directional))
    for order in orders:
        accepted = []
        for index in order:
            edge = directional[index]
            if longest(common[0] + accepted + [edge]) is not None:
                accepted.append(edge)
        signature = tuple(sorted(accepted))
        if signature not in seen_seeds:
            seeds.append((common[0] + accepted, common[1][:]))
            seen_seeds.add(signature)

    heap = []
    visited = set()
    serial = 0

    def enqueue(graphs, depth):
        nonlocal serial
        signature = (tuple(sorted(set(graphs[0]))), tuple(sorted(set(graphs[1]))))
        if signature in visited:
            return
        visited.add(signature)
        pos = coordinates(graphs)
        if pos is None:
            return
        key, collisions, pos = consider(pos)
        adjusted = project_near(pos, graphs)
        alt_key, alt_collisions, adjusted = consider(adjusted)
        # Branch on a candidate satisfying the linear graph. Projection is only
        # accepted above when those inequalities remain feasible.
        if alt_key > key:
            key, collisions, pos = alt_key, alt_collisions, adjusted
        serial += 1
        heapq.heappush(heap, (-key[0], -key[1], len(collisions), depth,
                              serial, graphs, pos, collisions))

    for seed in seeds:
        enqueue(seed, 0)
    expanded = 0
    generated = 0
    while heap and expanded < 100 and generated < 400:
        if best_key[0] >= 1.0 - 1e-12:
            break
        _, _, _, depth, _, graphs, pos, collisions = heapq.heappop(heap)
        expanded += 1
        if not collisions:
            continue
        _, a, b = collisions[0]
        alternatives = []
        for axis in range(2):
            distance = (sizes[a][axis] + sizes[b][axis]) / 2 + eps
            for u, v in ((a, b), (b, a)):
                edge = (u, v, distance)
                if edge not in graphs[axis]:
                    alternatives.append((distance - (pos[axis][v] - pos[axis][u]),
                                         axis, edge))
        alternatives.sort()
        for _, axis, edge in alternatives:
            child = [graphs[0][:], graphs[1][:]]
            child[axis].append(edge)
            enqueue(child, depth + 1)
            generated += 1
            if best_key[0] >= 1.0 - 1e-12:
                break

    result = {}
    for i, obj in enumerate(objects):
        z = min(max(sizes[i][2] / 2, heights[i]), room[2] - sizes[i][2] / 2)
        result[obj['id']] = [float(best_pos[0][i]), float(best_pos[1][i]), float(z)]
    return result
# EVOLVE-BLOCK-END
