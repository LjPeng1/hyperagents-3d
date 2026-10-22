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
    z = [min(max(size[i][2] / 2, z[i]), room[2] - size[i][2] / 2) for i in range(n)]
    source = n
    count = n + 1
    eps = 1e-7
    common = [[], []]
    for k in range(2):
        for i in range(n):
            half = size[i][k] / 2
            common[k].extend([(source, i, half), (i, source, half - room[k])])
            p = parent[i]
            if p is not None:
                allowance = (size[p][k] - size[i][k]) / 2
                common[k].extend([(p, i, -allowance), (i, p, -allowance)])
    orders = []
    near = []
    for r in scene.get('relations', []):
        a, b = ids[r['a']], ids[r['b']]
        if r['type'] == 'left_of':
            orders.append((a, b, (size[a][0] + size[b][0]) / 2 + float(r.get('gap', 0))))
        elif r['type'] == 'near':
            near.append((a, b, float(r['max_distance'])))
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)
             if abs(z[a] - z[b]) < (size[a][2] + size[b][2]) / 2 - 1e-9]

    def closure(edges):
        d = [[-math.inf] * count for _ in range(count)]
        for i in range(count):
            d[i][i] = 0.0
        for a, b, gap in edges:
            d[a][b] = max(d[a][b], gap)
        for k in range(count):
            row = d[k]
            for i in range(count):
                base = d[i][k]
                if base == -math.inf:
                    continue
                target = d[i]
                for j in range(count):
                    value = base + row[j]
                    if value > target[j]:
                        target[j] = value
            if any(d[i][i] > 1e-9 for i in range(count)):
                return None
        return d

    def prepare(graphs, hard):
        g = [graphs[0][:], graphs[1][:]]
        matrices = [closure(g[0]), closure(g[1])]
        if any(d is None for d in matrices):
            return None
        if hard:
            for _ in range(5):
                additions = [[], []]
                for a, b, radius in near:
                    if radius < 0:
                        return None
                    minimum = [max(0.0, d[a][b], d[b][a]) for d in matrices]
                    if math.hypot(*minimum) > radius + 1e-8:
                        return None
                    for k in range(2):
                        limit = math.sqrt(max(0.0, radius * radius - minimum[1-k] ** 2)) + 1e-9
                        for u, v in ((a, b), (b, a)):
                            if -limit > matrices[k][u][v] + 1e-10:
                                additions[k].append((u, v, -limit))
                if not any(additions):
                    break
                for k in range(2):
                    if additions[k]:
                        g[k].extend(additions[k])
                        matrices[k] = closure(g[k])
                        if matrices[k] is None:
                            return None
        pos = [[(d[source][i] - d[i][source]) / 2 for i in range(n)] for d in matrices]
        return g, matrices, pos

    def assess(pos):
        supported = 0
        for i, p in enumerate(parent):
            if p is None:
                supported += abs(z[i] - size[i][2] / 2) <= 1e-8
            else:
                supported += (abs(z[i] - z[p] - (size[p][2] + size[i][2]) / 2) <= 1e-8
                              and all(abs(pos[k][i] - pos[k][p]) <= (size[p][k] - size[i][k]) / 2 + 1e-8 for k in range(2)))
        collisions = []
        for a, b in pairs:
            penetration = [(size[a][k] + size[b][k]) / 2 - abs(pos[k][a] - pos[k][b]) for k in range(2)]
            if min(penetration) > 1e-8:
                collisions.append((penetration[0] * penetration[1], a, b))
        violations = [max(0.0, gap - pos[0][b] + pos[0][a]) for a, b, gap in orders]
        violations.extend(max(0.0, math.hypot(pos[0][a] - pos[0][b], pos[1][a] - pos[1][b]) - radius) for a, b, radius in near)
        fraction = sum(v <= 1e-8 for v in violations) / len(violations) if violations else 1.0
        score = (0.55 * supported / n + 0.45 * (1 - len(collisions) / max(1, n * (n-1) // 2))) * (0.6 + 0.4 * fraction)
        return (score, -sum(violations), -sum(t[0] for t in collisions)), collisions

    best = [[room[k] / 2 for _ in objects] for k in range(2)]
    best_key = (-math.inf, -math.inf, -math.inf)

    def consider(pos):
        nonlocal best, best_key
        pos = [[min(room[k] - size[i][k] / 2, max(size[i][k] / 2, pos[k][i])) for i in range(n)] for k in range(2)]
        key, collisions = assess(pos)
        if key > best_key:
            best_key = key
            best = [axis[:] for axis in pos]
        return key, collisions, pos

    consider(best)

    def project(pos, graphs):
        if not near:
            return pos
        work = [axis[:] + [0.0] for axis in pos]
        linear = [[0.0] * len(g) for g in graphs]
        disks = [[0.0, 0.0] for _ in near]
        for iteration in range(240):
            movement = 0.0
            for k, edges in enumerate(graphs):
                v = work[k]
                for index, (a, b, gap) in enumerate(edges):
                    if a == b:
                        continue
                    old = linear[k][index]
                    av = v[a] + old if a != source else 0.0
                    bv = v[b] - old if b != source else 0.0
                    delta = max(0.0, gap - bv + av) / (int(a != source) + int(b != source))
                    if a != source:
                        movement = max(movement, abs(av - delta - v[a]))
                        v[a] = av - delta
                    if b != source:
                        movement = max(movement, abs(bv + delta - v[b]))
                        v[b] = bv + delta
                    linear[k][index] = delta
            for index, (a, b, radius) in enumerate(near):
                if a == b:
                    continue
                av = [work[k][a] + disks[index][k] for k in range(2)]
                bv = [work[k][b] - disks[index][k] for k in range(2)]
                delta = [av[k] - bv[k] for k in range(2)]
                distance = math.hypot(*delta)
                factor = max(0.0, 1 - max(0.0, radius - 1e-9) / distance) / 2 if distance else 0.0
                for k in range(2):
                    correction = delta[k] * factor
                    na, nb = av[k] - correction, bv[k] + correction
                    movement = max(movement, abs(na - work[k][a]), abs(nb - work[k][b]))
                    work[k][a], work[k][b] = na, nb
                    disks[index][k] = correction
            if movement < 1e-11:
                break
        if any(work[k][b] - work[k][a] < gap - 1e-9 for k in range(2) for a, b, gap in graphs[k]):
            return pos
        return [axis[:n] for axis in work]

    seeds = []
    signatures = set()
    full = [(a, b, gap + eps) for a, b, gap in orders]
    for ordering in (list(range(len(full))), list(reversed(range(len(full)))), sorted(range(len(full)), key=lambda i: full[i][2])):
        accepted = []
        for index in ordering:
            if closure(common[0] + accepted + [full[index]]) is not None:
                accepted.append(full[index])
        sig = tuple(sorted(accepted))
        if sig not in signatures:
            signatures.add(sig)
            seeds.append([common[0] + accepted, common[1][:]])

    serial = 0
    for hard, budget in ((True, 100), (False, 60)):
        if best_key[0] >= 1 - 1e-12:
            break
        heap = []
        visited = set()

        def enqueue(graphs, depth):
            nonlocal serial
            signature = tuple(tuple(sorted(set(g))) for g in graphs)
            if signature in visited:
                return
            visited.add(signature)
            prepared = prepare(graphs, hard)
            if prepared is None:
                return
            g, matrices, pos = prepared
            key, collisions, pos = consider(pos)
            if best_key[0] >= 1 - 1e-12:
                return
            alt_key, alt_collisions, adjusted = consider(project(pos, g))
            if alt_key > key:
                key, collisions, pos = alt_key, alt_collisions, adjusted
            serial += 1
            heapq.heappush(heap, (-key[0], -key[1], len(collisions), depth, serial, g, matrices, pos, collisions))

        for seed in seeds:
            enqueue(seed, 0)
            if best_key[0] >= 1 - 1e-12:
                break
        expanded = 0
        while heap and expanded < budget and best_key[0] < 1 - 1e-12:
            _, _, _, depth, _, graphs, matrices, pos, collisions = heapq.heappop(heap)
            expanded += 1
            if not collisions:
                continue
            selected = None
            selected_key = None
            for area, a, b in collisions:
                alternatives = []
                for k in range(2):
                    distance = (size[a][k] + size[b][k]) / 2 + eps
                    for u, v in ((a, b), (b, a)):
                        if matrices[k][v][u] + distance <= 1e-9:
                            alternatives.append((distance - pos[k][v] + pos[k][u], k, (u, v, distance)))
                rank = (len(alternatives), -area, a, b)
                if selected_key is None or rank < selected_key:
                    selected_key, selected = rank, alternatives
            for _, k, edge in sorted(selected or []):
                child = [graphs[0][:], graphs[1][:]]
                child[k].append(edge)
                enqueue(child, depth + 1)
                if best_key[0] >= 1 - 1e-12:
                    break

    return {o['id']: [float(best[0][i]), float(best[1][i]), float(z[i])] for i, o in enumerate(objects)}
# EVOLVE-BLOCK-END
