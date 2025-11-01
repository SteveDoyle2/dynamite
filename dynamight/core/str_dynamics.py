from itertools import count
import matplotlib.pyplot as plt
import numpy as np


def _get_iedges(iedge: int, nedges: int) -> tuple[int, int, int]:
    """
    get ith value of looping list with repeating
    so:
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # iedge=0
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]  # iedge=1
    [2, 3, 4, 5, 6, 7, 8, 9, 0, 1]  # iedge=2
    [3, 4, 5, 6, 7, 8, 9, 0, 1, 2]  # iedge=4
    ...
    because that's a pain, we basically do this...
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,]
    """
    values = np.arange(nedges)
    i2 = values[(iedge+1) % nedges]
    i3 = values[(iedge+2) % nedges]
    i4 = values[(iedge+3) % nedges]
    return i2, i3, i4


def reduce_n_minus1_sided_thickness(
        points: np.ndarray,
        thickness: float,
        total_mass: float,
        normal_sign: float=1.0) -> tuple[np.ndarray, np.ndarray, list[np.ndarray],
                                         float, np.ndarray, np.ndarray]:
    npoints = len(points)
    nedges = npoints - 1
    points2, points3 = _get_internal_line_points(points, thickness)
    assert nedges == 3

    # reorg points2 to match the order of points
    i0 = npoints - 1   # TODO: is this always right?
    ii = np.arange(npoints)
    ii2 = np.hstack([ii, ii])
    ii2 = ii2[i0:i0+npoints]
    points2 = points2[ii2]
    del points3

    normal = np.array([1., 0., 0.])
    points_external = points.copy()
    points_internal = points2.copy()
    points_internal[0, :] = points_external[0, :] + normal*thickness*normal_sign
    points_internal[-1, :] = points_external[-1, :] - normal*thickness*normal_sign
    points_quads, area, total_area = get_area_quads_from_internal_external_points(
        points_internal, points_external, nedges)

    narea = len(area)
    scale = area / total_area
    mass = scale * total_mass

    # find the total cg of all sections
    cg = np.zeros((narea, 3))
    inertia = np.zeros((narea, 6))
    for iarea, pointsi, areai, massi in zip(count(), points_quads, area, mass):
        areai, cgi, inertiai = reduce_area_nsm(
            pointsi, massi,
            ax=None, num_interp=10, nround=6, add_data_to_plots=False)
        cg[iarea, :] = cgi
        inertia[iarea, :] = inertiai

    cg_total, inertia_total = combine_area_based_mass_cg_inertia(
        area, mass, cg, inertia)
    return points_internal, points_external, points_quads, total_area, cg_total, inertia_total


def get_area_quads_from_internal_external_points(points_internal: np.ndarray,
                                                 points_external: np.ndarray,
                                                 nedges: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # get the points/area by section
    points_quads = []
    area = np.zeros(nedges)
    for iedge in range(nedges):
        p1 = points_external[iedge, :]
        p2 = points_external[iedge+1, :]
        p1_prime = points_internal[iedge, :]
        p2_prime = points_internal[iedge+1, :]
        pointsi = np.vstack([p1, p1_prime, p2_prime, p2])
        a = p1 - p2_prime
        b = p2 - p1_prime
        areai = 0.5 * np.linalg.norm(np.cross(a, b))
        points_quads.append(pointsi)
        area[iedge] = areai
    total_area = area.sum()
    return points_quads, area, total_area


def combine_area_based_mass_cg_inertia(
        area: np.ndarray,
        mass: np.ndarray,
        cg: np.ndarray,
        inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert len(area) == len(mass)
    assert len(area) == len(cg)
    assert len(area) == len(inertia)
    total_mass = mass.sum()
    #cg_mass = (cg * mass[:, np.newaxis]) / total_mass
    cg_total = (cg * mass[:, np.newaxis]).sum(axis=0) / total_mass

    dxyz = cg - cg_total[np.newaxis, :]
    dx = dxyz[:, 0]
    dy = dxyz[:, 1]
    dz = dxyz[:, 2]
    ixx = (inertia[:, 0] + area * (dy**2 + dz**2)).sum()
    iyy = (inertia[:, 1] + area * (dx**2 + dz**2)).sum()
    izz = (inertia[:, 2] + area * (dx**2 + dy**2)).sum()
    ixy = (inertia[:, 3] + area * (dx*dy)).sum()
    ixz = (inertia[:, 4] + area * (dx*dz)).sum()
    iyz = (inertia[:, 5] + area * (dy*dz)).sum()

    inertia_total = np.array([ixx, iyy, izz, ixy, ixz, iyz])
    # print('cgs\n', cg)
    # print('cg_mass\n', cg_mass)
    # print('cg-total', cg_total)
    # print('dxyz', dxyz)
    # inertia = get_inertia(dmass, xyz, cg_total, nround=nround)
    return cg_total, inertia_total


def _get_internal_line_points(points: np.ndarray,
                              thickness: float) -> tuple[np.ndarray, np.ndarray]:
    k = np.array([0., 0., 1.])
    nedges = len(points)
    points2 = points.copy()
    for iedge, p1 in enumerate(points):
        i2, i3, i4 = _get_iedges(iedge, nedges)
        p2 = points[i2]
        p3 = points[i3]

        i21 = p2 - p1
        i32 = p3 - p2
        j21 = np.cross(k, i21)
        j32 = np.cross(k, i32)
        j21 /= np.linalg.norm(j21)
        j32 /= np.linalg.norm(j32)

        p1_prime1 = p1 + j21 * thickness
        p2_prime1 = p2 + j21 * thickness

        p2_prime2 = p2 + j32 * thickness
        p3_prime2 = p3 + j32 * thickness
        pint2 = line_line_intersection(
            p1_prime1, p2_prime1,
            p2_prime2, p3_prime2)
        points2[iedge, :] = pint2
    # assert len(points2) == 4
    points3 = np.vstack([points2, points2[0, :]])
    return points2, points3


def reduce_line_nsm(points,
                    thickness: float,
                    total_mass: float,
                    ax: plt.Axes,
                    num_interp: int=360,
                    nround: int=6) -> tuple[np.ndarray, np.ndarray]:
    """
    Handles general shapes. Use more points for curved shapes.

    TODO: Can't do a 1D rod, do that with an area.
    """
    ts = np.linspace(0., 1., num=num_interp, endpoint=False)
    nedges = len(points)
    points2, points3 = _get_internal_line_points(points, thickness)
    ax.plot(points[:, 0], points[:, 1], marker='s', linestyle='-', label='outer', color='C1')
    ax.plot(points3[:, 0], points3[:, 1], '--x', color='C0')

    xyz_list = []
    lengths = []
    nx = len(ts)
    short_length = 0.
    for iedge in range(nedges):
        p1 = points3[iedge, :]
        p2 = points3[iedge+1, :]
        dx = p2 - p1
        lengthi = np.linalg.norm(dx)
        xyzi = p1 + ts[:, np.newaxis] * dx[np.newaxis, :]
        lengthii = lengthi/nx
        length_arrayi = np.ones(nx) * lengthii
        assert len(length_arrayi) == len(xyzi), (len(length_arrayi), len(xyzi))
        lengths.append(length_arrayi)
        short_length += lengthi
        xyz_list.append(xyzi)
        #dmasses.append(dmass)
        # ax.plot(xyzi[:, 0], xyzi[:, 1], marker='o')
    length = np.hstack(lengths)
    total_length = sum(length)
    xyz = np.vstack(xyz_list)
    # assert len(length) == 40, length
    assert len(xyz) == len(length), (len(xyz), len(length))

    cg1 = np.mean(points, axis=0)
    cg2 = np.mean(points2, axis=0)  # internal points
    # cg3 = np.mean(points3, axis=0) # don't use this; tacks on another point to close the quad
    #dmass = total_mass * length / total_length
    dmass = total_mass * length / short_length
    # assert dmass.sum() == total_mass, (dmass.sum(), total_mass)
    cg_total = (xyz * dmass[:, np.newaxis]).sum(axis=0) / total_mass
    cg_total = cg_total.round(nround)

    print(f"cg_total = {cg_total}")
    ax.scatter(cg_total[0], cg_total[1], marker='o', label='cg', color='C2')
    ax.scatter(cg2[0], cg2[1], marker='o', label='cg-1/ax2', color='C1')
    # ax.scatter(cg1[0], cg1[1], marker='o', label='cg-1', color='C0')
    # ax.scatter(cg3[0], cg3[1], marker='o', label='cg-3')
    inertia = get_inertia(dmass, xyz, cg_total, nround=nround)
    ixx, iyy, izz, ixy, ixz, iyz = inertia

    fig = ax.get_figure()
    fig.suptitle(f'line smear; mass={total_mass:g} cg=[{cg_total[0]:g}, {cg_total[1]:g}, {cg_total[2]:g}]\n'
                 f'inertia=[{ixx:g}, {ixy:g}, {iyy:g}, {ixz:g}, {iyz:g}, {izz:g}]')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return cg_total, inertia


def line_line_intersection(p1, p2, p3, p4):
    """https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection"""
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    x4, y4, z4 = p4
    tnum = np.array([
        [x1 - x3, x3 - x4],
        [y1 - y3, y3 - y4],
    ])
    tdenom = np.array([
        [x1 - x2, x3 - x4],
        [y1 - y2, y3 - y4],
    ])
    t = np.linalg.det(tnum) / np.linalg.det(tdenom)

    # unum = np.array([
    #     [x1 - x2, x1 - x3],
    #     [y1 - y2, y1 - y3],
    # ])
    # udenom = np.array([
    #     [x1 - x2, x3 - x4],
    #     [y1 - y2, y3 - y4],
    # ])
    # u = np.linalg.det(unum) / np.linalg.det(udenom)
    pa = p1 + (p2 - p1) * t
    # pb = p3 + (p4 - p3) * u
    return pa


def reduce_area_nsm(points: np.ndarray,
                    mass: float,
                    ax: plt.Axes,
                    num_interp: int=40,
                    nround: int=6,
                    add_data_to_plots: bool=True) -> tuple[float, np.ndarray, np.ndarray]:
    """TODO: handle general shapes"""
    dab = 2.0 / num_interp
    t0 = -1 + dab/2
    t1 = 1 - dab/2

    p1 = points[0, :][np.newaxis, :]
    p2 = points[1, :][np.newaxis, :]
    p3 = points[2, :][np.newaxis, :]
    p4 = points[3, :][np.newaxis, :]
    xs = np.linspace(t0, t1, num=num_interp, endpoint=True)

    a, b = np.meshgrid(xs, xs)
    a = a.flatten()
    b = b.flatten()
    ab = np.column_stack([a, b])

    npoints = len(ab)
    # assert npoints > 5, npoints
    ab1 = np.column_stack([ab[:, 0]-dab/2, ab[:, 1]-dab/2])
    ab2 = np.column_stack([ab[:, 0]+dab/2, ab[:, 1]-dab/2])
    ab3 = np.column_stack([ab[:, 0]+dab/2, ab[:, 1]+dab/2])
    ab4 = np.column_stack([ab[:, 0]-dab/2, ab[:, 1]+dab/2])

    # ax.plot(ab1[:, 0], ab1[:, 1], marker='o', label='p1')
    # ax.plot(ab2[:, 0], ab2[:, 1], marker='o', label='p2')
    # ax.plot(ab3[:, 0], ab3[:, 1], marker='o', label='p3')
    # ax.plot(ab4[:, 0], ab4[:, 1], marker='o', label='p4')

    # shape functions for p1
    # abss = np.column_stack([ab1, ab2, ab3, ab4])
    N1 = get_nquad(ab1)
    pa = N1 @ points

    # shape functions for p2
    N2 = get_nquad(ab2)
    pb = N2 @ points

    # shape functions for p3
    N3 = get_nquad(ab3)
    pc = N3 @ points

    # shape functions for p4
    N4 = get_nquad(ab4)
    assert len(N2) == npoints
    pd = N4 @ points

    # ax.plot(pd[:, 0], pd[:, 1], 'o')
    p31 = pc - pa
    p42 = pd - pb
    axb = np.cross(p31, p42, axis=1)
    assert axb.shape == p31.shape, (axb.shape, p31.shape)
    area = 0.5 * np.linalg.norm(axb, axis=1)

    area_total = area.sum()
    debug = False
    if debug:
        print('ab = ', abss[0, :])
        print('N1 = ', N1[0, :])
        print('N2 = ', N2[0, :])
        print('N3 = ', N3[0, :])
        print('N4 = ', N4[0, :])
        print('pa0 = ', pa[0, :])
        print('pb0 = ', pb[0, :])
        print('pc0 = ', pc[0, :])
        print('pd0 = ', pd[0, :])
        print('area0 = ', area[0])
        print(f'area_total = {area_total}')
        print('mass = ', mass)
    assert len(area) == len(axb), len(area)
    xyz = (pa + pb + pc + pd) / 4

    dmass = mass * area / area_total

    cg_total = (p1 + p2 + p3 + p4) / 4
    inertia = get_inertia(dmass, xyz, cg_total, nround=nround)
    ixx, iyy, izz, ixy, ixz, iyz = inertia

    cg_total = cg_total.flatten()
    if add_data_to_plots:
        fig = ax.get_figure()
        ax.plot(cg_total[0], cg_total[1], 'o', label='cg')
        fig.suptitle(f'area smear; mass={mass:g} cg=[{cg_total[0]:g}, {cg_total[1]:g}, {cg_total[2]:g}]\n'
                     f'inertia=[{ixx:g}, {ixy:g}, {iyy:g}, {ixz:g}, {iyz:g}, {izz:g}]')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    assert len(cg_total) == 3, cg_total
    return area_total, cg_total, inertia


def get_inertia(dmass: np.ndarray,
                xyz: np.ndarray,
                cg_total: np.ndarray,
                nround: int=6) -> np.ndarray:
    dxyz = xyz - cg_total
    dx = dxyz[:, 0]
    dy = dxyz[:, 1]
    dz = dxyz[:, 2]
    ixxv = dmass * (dy ** 2 + dz ** 2)
    iyyv = dmass * (dx ** 2 + dz ** 2)
    izzv = dmass * (dx ** 2 + dy ** 2)
    ixyv = dmass * dx * dy
    iyzv = dmass * dy * dz
    ixzv = dmass * dx * dz
    ixx = ixxv.sum()
    iyy = iyyv.sum()
    izz = izzv.sum()
    ixz = ixzv.sum()
    iyz = iyzv.sum()
    ixy = ixyv.sum()
    # print(f'dmass = {dmass[0]}')
    # print(f'dxyz = {dxyz[0, :]}')
    # print(f'ixx={ixx:g}; iyy={iyy:g}; izz={izz:g}')
    # print(f'ixy={ixy:g}; iyz={iyz:g}; ixz={ixz:g}')
    inertia = np.array([ixx, iyy, izz, ixy, ixz, iyz])
    inertia = np.round(inertia, nround)
    return inertia


def get_nquad(ab: np.ndarray) -> np.ndarray:
    N1 = (1 - ab[:, 0]) * (1 - ab[:, 1]) / 4
    N2 = (1 + ab[:, 0]) * (1 - ab[:, 1]) / 4
    N3 = (1 + ab[:, 0]) * (1 + ab[:, 1]) / 4
    N4 = (1 - ab[:, 0]) * (1 + ab[:, 1]) / 4
    Ns = np.column_stack([N1, N2, N3, N4])
    return Ns


if __name__ == '__main__':
    main()
