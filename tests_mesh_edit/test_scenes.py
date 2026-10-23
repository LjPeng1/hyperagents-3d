import numpy as np

from portable.mesh_edit import scenes


def test_asset_families_do_not_cross_split():
    sets = [set(assets) for assets in scenes.SPLITS.values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i+1, 3))
    families = [{name.split('_')[0] for name in assets} for assets in scenes.SPLITS.values()]
    assert all(not families[i] & families[j] for i in range(3) for j in range(i+1, 3))


def test_deterministic_corruption_preserves_protected_region_and_has_no_reference_in_public():
    first = scenes.get_cases('calibration')
    second = scenes.get_cases('calibration')
    assert first == second and len(first) == 6
    for case in first:
        public = case['public']
        assert public['id'].startswith('case_')
        assert not any(split in public['id'] for split in scenes.SPLITS)
        assert set(public) == {'id', 'task', 'vertices', 'faces', 'editable', 'noise_hint'}
        noisy, clean = np.asarray(public['vertices']), np.asarray(case['reference'])
        mask = np.asarray(public['editable'])
        assert mask.any() and (~mask).any()
        assert np.array_equal(noisy[~mask], clean[~mask])
        assert np.linalg.norm(noisy[mask]-clean[mask]) > 0
        assert np.isfinite(noisy).all()


def test_fresh_rotation_and_vertex_permutation_preserve_intrinsic_mesh():
    a = scenes.make_case('head_generic', 'normal_noise', 11, 'a')
    b = scenes.make_case('head_generic', 'normal_noise', 12, 'b')
    def triangle_areas(case):
        v = np.asarray(case['reference']); f = np.asarray(case['public']['faces'])
        return np.sort(np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1))
    assert not np.allclose(a['reference'], b['reference'])
    assert np.allclose(triangle_areas(a), triangle_areas(b))
