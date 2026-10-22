"""Initial general heuristic. Only the marked block is evolved."""
# EVOLVE-BLOCK-START
def solve(scene):
    import math
    import heapq

    objects = scene['objects']
    n = len(objects)
    if not n:
        return {}
    anchor = n
    ids = {o['id']: i for i, o in enumerate(objects)}
    sizes = [list(map(float, o['size'])) for o in objects]
    room = list(map(float, scene['room']))
    parents = [None if o['support'] is None else ids[o['support']] for o in objects]
    relations = scene.get('relations', [])
    eps = 1e-8
    clearance = 1e-7
    z = [None] * n
    order = []
    pending = set(range(n))
    while pending:
        ready = sorted(i for i in pending if parents[i] is None or z[parents[i]] is not None)
        if not ready:
            raise ValueError('cyclic or missing support')
        for i in ready:
            p = parents[i]
            z[i] = sizes[i][2] / 2 if p is None else z[p] + (sizes[p][2] + sizes[i][2]) / 2
            order.append(i)
            pending.remove(i)

    def coordinates(edges):
        d = [0.0] * (n + 1)
        for _ in range(n + 1):
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
            graphs[axis].extend(((anchor, i, half), (i, anchor, half - room[axis])))
            p = parents[i]
            if p is not None:
                allowance = (sizes[p][axis] - sizes[i][axis]) / 2
                graphs[axis].extend(((p, i, -allowance), (i, p, -allowance)))
    gx, gy = tuple(graphs[0]), tuple(graphs[1])

    def fallback():
        result = {}
        for i in order:
            p = parents[i]
            xy = [room[0] / 2, room[1] / 2] if p is None else result[objects[p]['id']][:2]
            pos = xy + [z[i]]
            for axis in range(3):
                half = sizes[i][axis] / 2
                pos[axis] = max(half, min(room[axis] - half, pos[axis]))
            result[objects[i]['id']] = pos
        return result

    if coordinates(gx) is None or coordinates(gy) is None:
        return fallback()
    left, near = [], []
    for rel in relations:
        a, b = ids[rel['a']], ids[rel['b']]
        if rel['type'] == 'left_of':
            left.append((a, b, (sizes[a][0] + sizes[b][0]) / 2 + float(rel.get('gap', 0.0))))
        elif rel['type'] == 'near':
            near.append((a, b, float(rel['max_distance'])))
    all_left = gx + tuple((a, b, d + clearance) for a, b, d in left)
    if coordinates(all_left) is not None:
        gx = all_left
    else:
        for a, b, d in left:
            candidate = gx + ((a, b, d + clearance),)
            if coordinates(candidate) is not None:
                gx = candidate

    def bound(a, b, radius):
        return ((a, b, -radius), (b, a, -radius))

    for a, b, radius in near:
        if radius >= 0:
            nx, ny = gx + bound(a, b, radius), gy + bound(a, b, radius)
            if coordinates(nx) is not None and coordinates(ny) is not None:
                gx, gy = nx, ny

    vertical_pairs = [(j, i) for i in range(n) for j in range(i)
                      if (sizes[i][2] + sizes[j][2]) / 2 - abs(z[i] - z[j]) > eps]
    total_pairs = max(1, n * (n - 1) // 2)

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
        fraction = 1.0 if not relations else max(0.0, 1.0 - (len(bad_left) + len(bad_near)) / len(relations))
        score = (1.0 - 0.45 * len(collisions) / total_pairs) * (0.6 + 0.4 * fraction)
        return score, collisions, bad_left, bad_near

    def project_disks(ex, ey, x, y):
        # Sparse halfspace projections and disk projections with Dykstra residuals.
        linear = []
        for axis, edges in enumerate((ex, ey)):
            for u, v, c in edges:
                coeff = {}
                if u != anchor:
                    coeff[axis * n + u] = -1.0
                if v != anchor:
                    k = axis * n + v
                    coeff[k] = coeff.get(k, 0.0) + 1.0
                terms = tuple((k, value) for k, value in coeff.items() if value)
                if terms:
                    linear.append((terms, c, sum(value * value for _, value in terms)))
                elif c > eps:
                    return None
        disks = [(a, b, radius) for a, b, radius in near if a != b and radius >= 0]
        q = list(x) + list(y)
        residual = [0.0] * len(linear)
        disk_residual = [[0.0, 0.0] for _ in disks]
        candidate = None
        for sweep in range(320):
            for index, (terms, c, norm) in enumerate(linear):
                old = residual[index]
                value = 0.0
                for k, coefficient in terms:
                    q[k] += old * coefficient
                    value += q[k] * coefficient
                step = max(0.0, (c - value) / norm)
                for k, coefficient in terms:
                    q[k] += step * coefficient
                residual[index] = -step
            for index, (a, b, radius) in enumerate(disks):
                rx, ry = disk_residual[index]
                ax, ay = q[a] + rx, q[n + a] + ry
                bx, by = q[b] - rx, q[n + b] - ry
                dx, dy = ax - bx, ay - by
                distance = math.hypot(dx, dy)
                factor = 0.0 if distance <= radius else (distance - radius) / (2.0 * distance)
                rx, ry = factor * dx, factor * dy
                q[a], q[n + a] = ax - rx, ay - ry
                q[b], q[n + b] = bx + rx, by + ry
                disk_residual[index] = [rx, ry]
            if sweep % 8 == 7 or sweep == 319:
                if not all(math.isfinite(value) for value in q):
                    return candidate
                feasible = all(sum(q[k] * coefficient for k, coefficient in terms) >= c - 1e-9
                               for terms, c, _ in linear)
                if feasible:
                    px, py = q[:n], q[n:]
                    assessment = assess(px, py)
                    if candidate is None or assessment[0] > candidate[2][0]:
                        candidate = (px, py, assessment)
                    if not assessment[3]:
                        return candidate
        return candidate

    heap, seen = [], set()
    serial = 0
    best_score = -1.0
    best_xy = None

    def remember(x, y, assessment):
        nonlocal best_score, best_xy
        if assessment[0] > best_score:
            best_score = assessment[0]
            best_xy = (list(x), list(y))

    def enqueue(ex, ey, level):
        nonlocal serial
        key = (frozenset(ex), frozenset(ey))
        if key in seen:
            return
        seen.add(key)
        x, y = coordinates(ex), coordinates(ey)
        if x is None or y is None:
            return
        assessment = assess(x, y)
        remember(x, y, assessment)
        serial += 1
        penalty = sum(len(items) for items in assessment[1:])
        heapq.heappush(heap, (-assessment[0], penalty, level, serial, ex, ey, x, y, assessment))

    enqueue(gx, gy, 0)
    expansions = 0
    projection_calls = 0
    while heap and expansions < 700 and best_score < 1.0 - 1e-12:
        _, _, level, _, ex, ey, x, y, assessment = heapq.heappop(heap)
        expansions += 1
        if not assessment[1] and assessment[3] and projection_calls < 16:
            projection_calls += 1
            projected = project_disks(ex, ey, x, y)
            if projected is not None:
                px, py, pa = projected
                remember(px, py, pa)
                if pa[0] > assessment[0]:
                    x, y, assessment = px, py, pa
            if best_score >= 1.0 - 1e-12:
                break
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
                    dx, dy = safe_radius * math.cos(angle), safe_radius * math.sin(angle)
                    branches.append((ex + bound(a, b, dx), ey + bound(a, b, dy)))
        for nx, ny in branches:
            enqueue(nx, ny, level + 1)
    if best_xy is None:
        return fallback()
    x, y = best_xy
    result = {}
    for i, obj in enumerate(objects):
        px = max(sizes[i][0] / 2, min(room[0] - sizes[i][0] / 2, x[i]))
        py = max(sizes[i][1] / 2, min(room[1] - sizes[i][1] / 2, y[i]))
        pz = max(sizes[i][2] / 2, min(room[2] - sizes[i][2] / 2, z[i]))
        result[obj['id']] = [float(px), float(py), float(pz)]
    return result
# EVOLVE-BLOCK-END
