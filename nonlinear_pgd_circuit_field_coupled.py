"""Nonlinear PGD solver for the field-circuit coupled MQS case.

This script follows the asymmetric nonlinear PGD strategy requested here:

* PGD enrichment is the outer loop; the truncation error is checked per mode.
* For fixed time function G, the extended space mode Y=[R; q] is solved with
  an inner nonlinear material iteration.
* For fixed Y, the time function G is solved using the material state frozen
  at the end of the space solve.  The time solve does not update mu(B).

The reference mesh, material law, winding coupling, voltage excitation, and
boundary condition are taken from ``D:\\MQS_nonlinear_circuitfieldcoupled``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve


MU0 = 4.0 * np.pi * 1e-7


def core_mu_r_from_b(b):
    b = np.asarray(b, dtype=float)
    mu_r = 3474.33 / (1.0 + 1.9131 * np.power(np.maximum(b, 0.0), 5.2911))
    return np.clip(mu_r, 1.0, 3474.33)


def read_mesh_files(case_dir: Path):
    gcoord = np.loadtxt(case_dir / "gcoord.txt")
    node = np.loadtxt(case_dir / "node.txt", dtype=int)
    mat = np.loadtxt(case_dir / "mat.txt")
    wd = np.loadtxt(case_dir / "wd.txt", dtype=int)
    line = np.loadtxt(case_dir / "line.txt", dtype=int)
    bc = np.loadtxt(case_dir / "bc.txt", dtype=int) - 1
    return gcoord, node, mat, wd, line, bc


def element_geometry(gcoord, node):
    p1 = gcoord[node[:, 0]]
    p2 = gcoord[node[:, 1]]
    p3 = gcoord[node[:, 2]]
    b = np.column_stack(
        (p2[:, 1] - p3[:, 1], p3[:, 1] - p1[:, 1], p1[:, 1] - p2[:, 1])
    )
    c = np.column_stack(
        (p3[:, 0] - p2[:, 0], p1[:, 0] - p3[:, 0], p2[:, 0] - p1[:, 0])
    )
    area2 = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]
    area = 0.5 * area2
    return b, c, area, area2


def assemble_reluctivity_from_mu(gcoord, node, b_geom, c_geom, area, mu_r):
    reluctivity = 1.0 / (np.asarray(mu_r, dtype=float) * MU0)
    local = (reluctivity / (4.0 * area))[:, None, None] * (
        b_geom[:, :, None] * b_geom[:, None, :]
        + c_geom[:, :, None] * c_geom[:, None, :]
    )
    rows = np.repeat(node, 3, axis=1).ravel()
    cols = np.tile(node, (1, 3)).ravel()
    data = local.reshape(node.shape[0], 9).ravel()
    return coo_matrix((data, (rows, cols)), shape=(gcoord.shape[0], gcoord.shape[0])).tocsr()


def assemble_weighted_reluctivity(gcoord, node, b_geom, c_geom, area, mu_r_time, weights):
    """Assemble int (sum_n weights[n]*nu_n) grad(N)^T grad(N) dOmega."""
    weights = np.asarray(weights, dtype=float)
    reluctivity_time = 1.0 / (mu_r_time * MU0)
    weighted_nu = reluctivity_time @ weights
    local = (weighted_nu / (4.0 * area))[:, None, None] * (
        b_geom[:, :, None] * b_geom[:, None, :]
        + c_geom[:, :, None] * c_geom[:, None, :]
    )
    rows = np.repeat(node, 3, axis=1).ravel()
    cols = np.tile(node, (1, 3)).ravel()
    data = local.reshape(node.shape[0], 9).ravel()
    return coo_matrix((data, (rows, cols)), shape=(gcoord.shape[0], gcoord.shape[0])).tocsr()


def assemble_conductivity_matrix(gcoord, node, mat):
    rows, cols, data = [], [], []
    for iel, elem_nodes in enumerate(node):
        x1, y1 = gcoord[elem_nodes[0]]
        x2, y2 = gcoord[elem_nodes[1]]
        x3, y3 = gcoord[elem_nodes[2]]
        area = 0.5 * ((y2 - y3) * (x1 - x3) - (y3 - y1) * (x3 - x2))
        if mat[iel] in (2, 7):
            sigma = 1e-13
        elif mat[iel] == 1:
            sigma = 1e-6
        else:
            sigma = 0.0
        elem = sigma * np.array(
            [[area / 6, area / 12, area / 12],
             [area / 12, area / 6, area / 12],
             [area / 12, area / 12, area / 6]]
        )
        for i in range(3):
            for j in range(3):
                rows.append(elem_nodes[i])
                cols.append(elem_nodes[j])
                data.append(elem[i, j])
    return coo_matrix((data, (rows, cols)), shape=(gcoord.shape[0], gcoord.shape[0])).tocsr()


def winding_nodes(wd, col):
    return wd[wd[:, col] > 0, col] - 1


def assemble_winding_mass(gcoord, node, mat):
    rows, cols, data = [], [], []
    for iel, elem_nodes in enumerate(node):
        if mat[iel] not in (3, 4, 6, 7):
            continue
        x1, y1 = gcoord[elem_nodes[0]]
        x2, y2 = gcoord[elem_nodes[1]]
        x3, y3 = gcoord[elem_nodes[2]]
        area = abs(0.5 * ((y2 - y3) * (x1 - x3) - (y3 - y1) * (x3 - x2)))
        elem = np.array(
            [[area / 6, area / 12, area / 12],
             [area / 12, area / 6, area / 12],
             [area / 12, area / 12, area / 6]]
        )
        for i in range(3):
            for j in range(3):
                rows.append(elem_nodes[i])
                cols.append(elem_nodes[j])
                data.append(elem[i, j])
    return coo_matrix((data, (rows, cols)), shape=(gcoord.shape[0], gcoord.shape[0])).tocsr()


def assemble_loop_coupling(gcoord, node, mat, wd, sr):
    base = assemble_winding_mass(gcoord, node, mat)
    nnode = gcoord.shape[0]
    c_loop = np.zeros((nnode, 2))
    for col, sign, winding in ((0, -1.0, 0), (1, 1.0, 1), (2, -1.0, 1), (3, 1.0, 0)):
        idx = winding_nodes(wd, col)
        turns, section = sr[winding]
        marker = np.zeros(nnode)
        marker[idx] = sign * turns / section
        c_loop[:, winding] += base @ marker
    return coo_matrix(c_loop).tocsr()


def current_density_display(wd, il, ih, nnode):
    current = np.zeros(nnode)
    current[winding_nodes(wd, 0)] = ih
    current[winding_nodes(wd, 1)] = il
    current[winding_nodes(wd, 2)] = il
    current[winding_nodes(wd, 3)] = ih
    return current


def element_b_magnitude(node, b_geom, c_geom, area2, a):
    ae = a[node]
    bx = np.sum(c_geom * ae, axis=1) / area2
    by = -np.sum(b_geom * ae, axis=1) / area2
    return np.sqrt(bx * bx + by * by)


def nodal_b_magnitude(node, b_geom, c_geom, area2, nnode, a):
    be = element_b_magnitude(node, b_geom, c_geom, area2, a)
    bn = np.zeros(nnode)
    counts = np.zeros(nnode)
    np.add.at(bn, node.ravel(), np.repeat(be, 3))
    np.add.at(counts, node.ravel(), 1)
    mask = counts > 0
    bn[mask] /= counts[mask]
    return bn


def mu_time_from_solution(node, mat, b_geom, c_geom, area2, av):
    nelem, nt = node.shape[0], av.shape[1]
    mu_time = np.ones((nelem, nt))
    core = mat == 1
    mu_time[core, :] = 1200.0
    for n in range(nt):
        elem_b = element_b_magnitude(node, b_geom, c_geom, area2, av[:, n])
        mu_time[core, n] = core_mu_r_from_b(elem_b[core])
    return mu_time


def reduced_free_indices(nnode, nloop, bc):
    free_a = np.ones(nnode, dtype=bool)
    free_a[bc] = False
    return np.concatenate([np.where(free_a)[0], np.arange(nnode, nnode + nloop)])


def build_constant_blocks(k_mat, c_loop, dt, resistance, inductance, lef):
    nnode = k_mat.shape[0]
    nloop = c_loop.shape[1]
    r = diags(np.asarray(resistance, dtype=float), format="csr")
    l = diags(np.asarray(inductance, dtype=float), format="csr")
    lef_mat = diags(np.asarray(lef, dtype=float), format="csr")
    z_ai = csr_matrix((nnode, nloop))
    z_ia = csr_matrix((nloop, nnode))
    d_mat = bmat([[k_mat, z_ai], [lef_mat @ c_loop.T, l]], format="csr")
    f_mat = bmat([[csr_matrix((nnode, nloop))], [csr_matrix(np.eye(nloop))]], format="csr")
    return r, l, lef_mat, d_mat, f_mat


def build_b_matrix(m_mat, k_mat, c_loop, dt, r, l, lef_mat):
    return bmat(
        [[dt * m_mat + k_mat, -dt * c_loop],
         [lef_mat @ c_loop.T, dt * r + l]],
        format="csr",
    )


def build_weighted_b_matrix(m_weighted, alpha_k, alpha_c, k_mat, c_loop, dt, r, l, lef_mat):
    """Build sum_n w[n]*B_n for w-dependent reluctivity and scalar constant blocks."""
    return bmat(
        [[dt * m_weighted + alpha_k * k_mat, -dt * alpha_c * c_loop],
         [alpha_c * (lef_mat @ c_loop.T), alpha_c * (dt * r + l)]],
        format="csr",
    )


def solve_space_mode_nonlinear(
    gcoord,
    node,
    mat,
    b_geom,
    c_geom,
    area,
    area2,
    k_mat,
    c_loop,
    dt,
    r,
    l,
    lef_mat,
    d_red,
    f_red,
    free_idx,
    voltage,
    g,
    modes_y_full,
    modes_g,
    y0_full,
    nonlinear_tol,
    nonlinear_max_iter,
    nonlinear_relaxation,
):
    """For fixed G, solve Y while updating nonlinear mu(B) only here."""
    nnode = gcoord.shape[0]
    nloop = c_loop.shape[1]
    nt = len(g)
    y_full = y0_full.copy()
    if np.linalg.norm(y_full) == 0.0:
        y_full[nnode:] = 1.0

    prev_av = np.zeros((nnode, nt))
    for y_i, g_i in zip(modes_y_full, modes_g):
        prev_av += y_i[:nnode, None] * g_i[None, :]

    alpha_d = float(np.dot(g[1:], g[:-1]))
    alpha_f = g @ voltage
    last_mu_time = None
    last_b_red = None
    err = np.inf

    for it in range(1, nonlinear_max_iter + 1):
        av_guess = prev_av + y_full[:nnode, None] * g[None, :]
        mu_time = mu_time_from_solution(node, mat, b_geom, c_geom, area2, av_guess)
        m_lhs = assemble_weighted_reluctivity(gcoord, node, b_geom, c_geom, area, mu_time, g * g)
        b_alpha = build_weighted_b_matrix(
            m_lhs, float(np.dot(g, g)), float(np.dot(g, g)),
            k_mat, c_loop, dt, r, l, lef_mat
        )
        lhs_red = b_alpha[free_idx][:, free_idx] - alpha_d * d_red
        rhs = dt * (f_red @ alpha_f)

        for y_i, g_i in zip(modes_y_full, modes_g):
            m_prev = assemble_weighted_reluctivity(gcoord, node, b_geom, c_geom, area, mu_time, g * g_i)
            b_prev = build_weighted_b_matrix(
                m_prev, float(np.dot(g, g_i)), float(np.dot(g, g_i)),
                k_mat, c_loop, dt, r, l, lef_mat
            )
            rhs -= (b_prev[free_idx][:, free_idx] - float(np.dot(g[1:], g_i[:-1])) * d_red) @ y_i[free_idx]

        y_red_new = spsolve(lhs_red.tocsc(), rhs)
        y_new_full = np.zeros(nnode + nloop)
        y_new_full[free_idx] = y_red_new
        if np.linalg.norm(y_new_full) == 0.0:
            raise RuntimeError("Computed a zero nonlinear PGD space mode.")
        y_relaxed = nonlinear_relaxation * y_new_full + (1.0 - nonlinear_relaxation) * y_full

        err = np.linalg.norm(y_relaxed - y_full) / max(np.linalg.norm(y_relaxed), 1e-30)
        y_full = y_relaxed
        last_mu_time = mu_time
        last_b_red = b_alpha[free_idx][:, free_idx].tocsr()
        if err < nonlinear_tol:
            break

    # Freeze the final material state consistent with the converged Y.
    av_final = prev_av + y_full[:nnode, None] * g[None, :]
    last_mu_time = mu_time_from_solution(node, mat, b_geom, c_geom, area2, av_final)
    return y_full, last_mu_time, it, err


def b_times_vector_at_step(m_mat, k_mat, c_loop, dt, r, l, lef_mat, y):
    nnode = k_mat.shape[0]
    a = y[:nnode]
    q = y[nnode:]
    top = dt * (m_mat @ a) + k_mat @ a - dt * (c_loop @ q)
    bottom = lef_mat @ (c_loop.T @ a) + (dt * r + l) @ q
    return np.concatenate([np.asarray(top).reshape(-1), np.asarray(bottom).reshape(-1)])


def solve_time_mode_frozen_material(
    gcoord,
    node,
    b_geom,
    c_geom,
    area,
    mu_time,
    k_mat,
    c_loop,
    dt,
    r,
    l,
    lef_mat,
    d_red,
    f_red,
    free_idx,
    voltage,
    y_full,
    modes_y_full,
    modes_g,
):
    """Solve G using the material state frozen by the nonlinear space solve."""
    nt = voltage.shape[0]
    g = np.zeros(nt)
    y_red = y_full[free_idx]
    d_y = float(y_red @ (d_red @ y_red))
    f_y = np.asarray(y_red @ f_red).reshape(-1)
    d_prev = [float(y_red @ (d_red @ y_i[free_idx])) for y_i in modes_y_full]

    b_self = np.zeros(nt)
    b_prev = [np.zeros(nt) for _ in modes_y_full]
    for n in range(nt):
        m_n = assemble_reluctivity_from_mu(gcoord, node, b_geom, c_geom, area, mu_time[:, n])
        by = b_times_vector_at_step(m_n, k_mat, c_loop, dt, r, l, lef_mat, y_full)
        b_self[n] = float(y_red @ by[free_idx])
        for j, y_i in enumerate(modes_y_full):
            bi = b_times_vector_at_step(m_n, k_mat, c_loop, dt, r, l, lef_mat, y_i)
            b_prev[j][n] = float(y_red @ bi[free_idx])

    rhs0 = dt * float(f_y @ voltage[0, :])
    for bp, g_i in zip(b_prev, modes_g):
        rhs0 -= bp[0] * g_i[0]
    g[0] = rhs0 / b_self[0]

    for n in range(1, nt):
        rhs = d_y * g[n - 1] + dt * float(f_y @ voltage[n, :])
        for bp, dp, g_i in zip(b_prev, d_prev, modes_g):
            rhs -= bp[n] * g_i[n] - dp * g_i[n - 1]
        g[n] = rhs / b_self[n]
    return g


def nonlinear_pgd_solve(
    gcoord,
    node,
    mat,
    b_geom,
    c_geom,
    area,
    area2,
    k_mat,
    c_loop,
    dt,
    r,
    l,
    lef_mat,
    d_mat,
    f_mat,
    free_idx,
    voltage,
    max_modes,
    trunc_tol,
    pgd_inner_tol,
    pgd_max_inner,
    nonlinear_tol,
    nonlinear_max_iter,
    nonlinear_relaxation,
):
    d_red = d_mat[free_idx][:, free_idx].tocsr()
    f_red = f_mat[free_idx, :].tocsr()
    nfull = gcoord.shape[0] + c_loop.shape[1]
    modes_y = []
    modes_g = []
    mode_info = []
    max_weight = 0.0
    g_init = voltage[:, 0] / max(np.linalg.norm(voltage[:, 0]), 1.0)

    for mode in range(1, max_modes + 1):
        g = g_init.copy()
        y = np.zeros(nfull)
        last_mu = None
        space_nl_iter = 0
        space_nl_err = np.inf
        inner_err = np.inf

        for inner in range(1, pgd_max_inner + 1):
            y_old = y.copy()
            g_old = g.copy()
            y, last_mu, space_nl_iter, space_nl_err = solve_space_mode_nonlinear(
                gcoord, node, mat, b_geom, c_geom, area, area2,
                k_mat, c_loop, dt, r, l, lef_mat, d_red, f_red, free_idx,
                voltage, g, modes_y, modes_g, y,
                nonlinear_tol, nonlinear_max_iter, nonlinear_relaxation,
            )
            g = solve_time_mode_frozen_material(
                gcoord, node, b_geom, c_geom, area, last_mu,
                k_mat, c_loop, dt, r, l, lef_mat, d_red, f_red, free_idx,
                voltage, y, modes_y, modes_g,
            )
            y_norm = np.linalg.norm(y)
            if y_norm == 0.0:
                raise RuntimeError("Computed a zero nonlinear PGD mode.")
            y = y / y_norm
            g = g * y_norm
            inner_err = max(
                np.linalg.norm(y - y_old) / max(np.linalg.norm(y), 1e-30),
                np.linalg.norm(g - g_old) / max(np.linalg.norm(g), 1e-30),
            )
            if inner_err < pgd_inner_tol:
                break

        weight = np.linalg.norm(g)
        max_weight = max(max_weight, weight)
        rel_weight = weight / max(max_weight, 1e-30)
        if inner_err > 10.0 * pgd_inner_tol and mode > 1:
            print(
                f"mode {mode:02d}: discarded because PGD inner iteration did not converge "
                f"(inner_err={inner_err:.3e})",
                flush=True,
            )
            break
        modes_y.append(y)
        modes_g.append(g)
        mode_info.append([mode, inner, space_nl_iter, inner_err, space_nl_err, weight, rel_weight])
        print(
            f"mode {mode:02d}: pgd_inner={inner:02d}, "
            f"space_nl_iter={space_nl_iter:02d}, inner_err={inner_err:.3e}, "
            f"space_nl_err={space_nl_err:.3e}, relative={rel_weight:.3e}",
            flush=True,
        )
        if mode > 1 and rel_weight < trunc_tol:
            break

    return np.column_stack(modes_y), np.column_stack(modes_g), np.asarray(mode_info)


def plot_flux(gcoord, node, line, values, out_path):
    triang = mtri.Triangulation(gcoord[:, 0], gcoord[:, 1], node)
    fig, ax = plt.subplots(figsize=(10, 8))
    tcf = ax.tripcolor(triang, values, shading="gouraud", cmap="jet")
    fig.colorbar(tcf, ax=ax, label="|B|")
    for i1, i2 in line:
        ax.plot([gcoord[i1, 0], gcoord[i2, 0]], [gcoord[i1, 1], gcoord[i2, 1]],
                color="black", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_node_history(time, fx, node_index, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(time, fx[node_index, :], linewidth=1.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("|B|")
    ax.set_title(f"Magnetic flux density at node {node_index}")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=Path(r"D:\MQS_nonlinear_circuitfieldcoupled"))
    parser.add_argument("--out-dir", type=Path, default=Path(r"D:\MQS_nonlinear_PGD_circuitfieldcoupled"))
    parser.add_argument("--max-modes", type=int, default=6)
    parser.add_argument("--trunc-tol", type=float, default=1e-5)
    parser.add_argument("--pgd-inner-tol", type=float, default=1e-5)
    parser.add_argument("--pgd-max-inner", type=int, default=12)
    parser.add_argument("--nonlinear-tol", type=float, default=1e-4)
    parser.add_argument("--nonlinear-max-iter", type=int, default=20)
    parser.add_argument("--nonlinear-relaxation", type=float, default=0.3)
    parser.add_argument("--probe-node", type=int, default=5209)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sr = [[980, 0.0988], [425, 0.07904]]
    model_depth = 0.4228
    time = np.arange(0.0, 0.020001, 0.0001)
    dt = 0.0001
    voltage = np.column_stack(
        (107.5e3 * np.cos(100.0 * np.pi * time),
         -46.0e3 * np.cos(100.0 * np.pi * time))
    )
    resistance = np.array([0.0, 0.0])
    inductance = np.array([0.0, 0.0])
    lef = np.array([model_depth, model_depth])

    gcoord, node, mat, wd, line, bc = read_mesh_files(args.case_dir)
    nnode = gcoord.shape[0]
    b_geom, c_geom, area, area2 = element_geometry(gcoord, node)

    print("assembling constant matrices...", flush=True)
    k_mat = assemble_conductivity_matrix(gcoord, node, mat)
    c_loop = assemble_loop_coupling(gcoord, node, mat, wd, sr)
    r, l, lef_mat, d_mat, f_mat = build_constant_blocks(k_mat, c_loop, dt, resistance, inductance, lef)
    free_idx = reduced_free_indices(nnode, c_loop.shape[1], bc)

    print("running nonlinear PGD enrichment...", flush=True)
    y_modes, g_modes, mode_info = nonlinear_pgd_solve(
        gcoord, node, mat, b_geom, c_geom, area, area2,
        k_mat, c_loop, dt, r, l, lef_mat, d_mat, f_mat, free_idx, voltage,
        args.max_modes, args.trunc_tol, args.pgd_inner_tol, args.pgd_max_inner,
        args.nonlinear_tol, args.nonlinear_max_iter, args.nonlinear_relaxation,
    )

    x = y_modes @ g_modes.T
    av = x[:nnode, :]
    currents = x[nnode:, :]

    print("computing outputs...", flush=True)
    fx = np.column_stack([
        nodal_b_magnitude(node, b_geom, c_geom, area2, nnode, av[:, i])
        for i in range(len(time))
    ])
    ic = np.column_stack([
        current_density_display(wd, currents[1, i], currents[0, i], nnode)
        for i in range(len(time))
    ])

    final_mu = mu_time_from_solution(node, mat, b_geom, c_geom, area2, av)
    core_mu = final_mu[mat == 1, :]
    mu_core_summary = np.column_stack((core_mu.min(axis=0), core_mu.mean(axis=0), core_mu.max(axis=0)))

    np.save(args.out_dir / "Av_pgd.npy", av)
    np.save(args.out_dir / "Fx_pgd.npy", fx)
    np.save(args.out_dir / "Ic_pgd.npy", ic)
    np.save(args.out_dir / "Ic_loop_pgd.npy", currents)
    np.save(args.out_dir / "Mu_core_pgd.npy", mu_core_summary)
    np.save(args.out_dir / "PGD_mode_info.npy", mode_info)
    np.savez(args.out_dir / "nonlinear_pgd_modes.npz", Y=y_modes, G=g_modes, time=time)

    for plot_idx, label in ((50, "0p005s"), (100, "0p010s")):
        if plot_idx < len(time):
            plot_flux(gcoord, node, line, fx[:, plot_idx],
                      args.out_dir / f"nonlinear_pgd_flux_step_{plot_idx}_{label}.png")
    if 0 <= args.probe_node < nnode:
        plot_node_history(time, fx, args.probe_node,
                          args.out_dir / f"nonlinear_pgd_flux_node_{args.probe_node}.png")

    print(f"saved nonlinear PGD results to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
