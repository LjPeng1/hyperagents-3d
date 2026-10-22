"""Initial general heuristic. Only the marked block is evolved."""
# EVOLVE-BLOCK-START
def solve(scene):
    import math
    import heapq

    objects = scene['objects']
    if not objects:
        return {}
    n = len(objects)
    anchor = n
    count = n + 1
    ids = {obj['id']: i for i, obj in enumerate(objects)}
    sizes = [list(map(float, obj['size'])) for obj in objects]
    room = list(map(float, scene['room']))
    parents = [None if obj['support'] is None else ids[obj['support']]
               for obj in objects]
    relations = scene.get('relations', [])
    eps = 1e-8
    clearance = 1e-7

    z = [None] * n
    order = []
    pending = set(range(n))
    while pending:
        ready = sorted(i for i in pending
                       if parents[i] is None or z[parents[i]] is not None)
        if not ready:
            raise ValueError('cyclic or missing support')
        for i in ready:
            p = parents[i]
            z[i] = sizes[i][2] / 2 if p is None else (
                z[p] + (sizes[p][2] + sizes[i][2]) / 2)
            order.append(i)
            pending.remove(i)

    # An edge (u, v, c) means coordinate[v] >= coordinate[u] + c.
    def coordinates(edges):
        d = [0.0] * count
        for iteration in range(count):
            changed = False
            for u, v, c in edges:
                value = d[u] + c
                if value > d[v] + 1e-10:
                    d[v] = value
                    changed = True
            if not changed:
                return [d[i] - d[anchor] for i in range(n)]
        return None

    graphs = [[], []]
    for axis in range(2):
        for i in range(n):
            half = sizes[i][axis] / 2
            graphs[axis].append((anchor, i, half))
            graphs[axis].append((i, anchor, half - room[axis]))
            p = parents[i]
            if p is not None:
                allowance = (sizes[p][axis] - sizes[i][axis]) / 2
                graphs[axis].append((p, i, -allowance))
                graphs[axis].append((i, p, -allowance))
    gx, gy = tuple(graphs[0]), tuple(graphs[1])

    def fallback():
        result = {}
        for i in order:
            p = parents[i]
            center = [room[0] / 2, room[1] / 2] if p is None else result[objects[p]['id']][:2]
            position = center + [z[i]]
            for axis in range(3):
                half = sizes[i][axis] / 2
                position[axis] = max(half, min(room[axis] - half, position[axis]))
            result[objects[i]['id']] = position
        return result

    if coordinates(gx) is None or coordinates(gy) is None:
        return fallback()

    left = []
    near = []
    for rel in relations:
        a, b = ids[rel['a']], ids[rel['b']]
        if rel['type'] == 'left_of':
            gap = float(rel.get('gap', 0.0))
            separation = (sizes[a][0] + sizes[b][0]) / 2 + gap
            left.append((a, b, separation))
        elif rel['type'] == 'near':
            near.append((a, b, float(rel['max_distance'])))

    # Prefer all directional relations when jointly feasible; otherwise retain
    # a deterministic feasible subset, with every relation still scored below.
    all_left = gx + tuple((a, b, distance + clearance)
                          for a, b, distance in left)
    if coordinates(all_left) is not None:
        gx = all_left
    else:
        for a, b, distance in left:
            candidate = gx + ((a, b, distance + clearance),)
            if coordinates(candidate) is not None:
                gx = candidate

    def bound(a, b, distance):
        return ((a, b, -distance), (b, a, -distance))

    # These bounds are necessary for circular XY proximity, but not sufficient.
    for a, b, radius in near:
        if radius < 0:
            continue
        nx, ny = gx + bound(a, b, radius), gy + bound(a, b, radius)
        if coordinates(nx) is not None and coordinates(ny) is not None:
            gx, gy = nx, ny

    vertical_pairs = []
    for i in range(n):
        for j in range(i):
            if (sizes[i][2] + sizes[j][2]) / 2 - abs(z[i] - z[j]) > eps:
                vertical_pairs.append((j, i))
    total_pairs = max(1, n * (n - 1) // 2)
    relation_count = len(relations)

    def assess(x, y):
        collisions = []
        for a, b in vertical_pairs:
            ox = (sizes[a][0] + sizes[b][0]) / 2 - abs(x[a] - x[b])
            oy = (sizes[a][1] + sizes[b][1]) / 2 - abs(y[a] - y[b])
            if ox > eps and oy > eps:
                collisions.append((min(ox, oy), a, b))
        bad_left = [(a, b, d) for a, b, d in left if x[b] - x[a] < d - eps]
        bad_near = []
        for a, b, radius in near:
            distance = math.hypot(x[a] - x[b], y[a] - y[b])
            if distance > radius + eps:
                bad_near.append((distance - radius, a, b, radius))
        fraction = 1.0 if not relation_count else max(
            0.0, 1.0 - (len(bad_left) + len(bad_near)) / relation_count)
        score = (1.0 - 0.45 * len(collisions) / total_pairs) * (0.6 + 0.4 * fraction)
        return score, collisions, bad_left, bad_near

    heap = []
    seen = set()
    serial = 0
    best_score = -1.0
    best_xy = None

    def enqueue(ex, ey, level):
        nonlocal serial, best_score, best_xy
        key = (frozenset(ex), frozenset(ey))
        if key in seen:
            return
        seen.add(key)
        x, y = coordinates(ex), coordinates(ey)
        if x is None or y is None:
            return
        assessment = assess(x, y)
        score, collisions, bad_left, bad_near = assessment
        if score > best_score:
            best_score = score
            best_xy = (x, y)
        serial += 1
        penalty = len(collisions) + len(bad_left) + len(bad_near)
        heapq.heappush(heap, (-score, penalty, level, serial, ex, ey, x, y, assessment))

    enqueue(gx, gy, 0)
    expansions = 0
    while heap and expansions < 700 and best_score < 1.0 - 1e-12:
        _, _, level, _, ex, ey, x, y, assessment = heapq.heappop(heap)
        expansions += 1
        _, collisions, bad_left, bad_near = assessment
        branches = []
        if collisions:
            _, a, b = max(collisions)
            for axis in range(2):
                separation = (sizes[a][axis] + sizes[b][axis]) / 2 + clearance
                for u, v in ((a, b), (b, a)):
                    edge = ((u, v, separation),)
                    branches.append((ex + edge, ey) if axis == 0 else (ex, ey + edge))
        elif bad_left:
            for a, b, distance in bad_left:
                branches.append((ex + ((a, b, distance + clearance),), ey))
        elif bad_near:
            _, a, b, radius = max(bad_near)
            if radius >= 0:
                safe_radius = max(0.0, radius - clearance)
                angles = [k * math.pi / 12 for k in range(7)]
                angles.append(math.atan2(abs(y[a] - y[b]), abs(x[a] - x[b])))
                if safe_radius > 0:
                    angles.append(math.acos(min(1.0, abs(x[a] - x[b]) / safe_radius)))
                    angles.append(math.asin(min(1.0, abs(y[a] - y[b]) / safe_radius)))
                for angle in angles:
                    dx = safe_radius * math.cos(angle)
                    dy = safe_radius * math.sin(angle)
                    branches.append((ex + bound(a, b, dx), ey + bound(a, b, dy)))
        for nx, ny in branches:
            enqueue(nx, ny, level + 1)

    if best_xy is None:
        return fallback()
    x, y = best_xy
    result = {}
    for i, obj in enumerate(objects):
        # Remove tiny numerical room-bound excursions from graph relaxation.
        px = max(sizes[i][0] / 2, min(room[0] - sizes[i][0] / 2, x[i]))
        py = max(sizes[i][1] / 2, min(room[1] - sizes[i][1] / 2, y[i]))
        result[obj['id']] = [float(px), float(py), float(z[i])]
    return result
# EVOLVE-BLOCK-END
