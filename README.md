# Nonlinear Field-Circuit Coupled MQS PGD Solver

This folder contains a Python implementation of the nonlinear PGD strategy
requested for the case in `D:\MQS_nonlinear_circuitfieldcoupled`.

## Nonlinear PGD Strategy

The extended unknown is:

```text
X(t) = [A(t); I(t)]
X(t) ~= sum_m Y_m G_m(t),  Y_m = [F_m; q_m]
```

For each PGD mode:

1. Fix the time function `G_m(t)`.
2. Solve the extended space mode `Y_m=[F_m; q_m]`.
3. During this space solve only, update the nonlinear core permeability:

```text
B = curl(A)
mu_r(B) = 3474.33 / (1 + 1.9131*B^5.2911)
```

4. Freeze the material state from the converged space solve.
5. Fix `Y_m` and solve `G_m(t)` using the frozen material state. No material
   update is performed in the time solve.
6. Save the mode and use the mode weight as the PGD truncation criterion.

This matches the intended asymmetric treatment:

```text
space equation updates nonlinear material
time equation uses the final frozen material from the space equation
outer loop checks PGD truncation error
```

## Same Settings as Reference Case

- Mesh, material tags, winding domains, and boundary nodes are read from
  `D:\MQS_nonlinear_circuitfieldcoupled`.
- The core is material tag `1`.
- Other magnetic domains use air permeability.
- Voltage excitation:
  - `U_h = 107.5e3*cos(100*pi*t)`
  - `U_l = -46.0e3*cos(100*pi*t)`
- Circuit settings:
  - `R_loop = [0, 0]`
  - `L_loop = [0, 0]`
  - `Lef_loop = [0.4228, 0.4228]`

## Run

```powershell
cd D:\MQS_nonlinear_PGD_circuitfieldcoupled
python -u .\nonlinear_pgd_circuit_field_coupled.py
```

Quick smoke test:

```powershell
python -u .\nonlinear_pgd_circuit_field_coupled.py --max-modes 1 --pgd-max-inner 2 --nonlinear-max-iter 2
```

## Outputs

- `Av_pgd.npy`: magnetic vector potential.
- `Fx_pgd.npy`: nodal magnetic flux density magnitude.
- `Ic_pgd.npy`: winding current-density display array.
- `Ic_loop_pgd.npy`: two coupled loop currents.
- `Mu_core_pgd.npy`: min/mean/max nonlinear core relative permeability per time step.
- `PGD_mode_info.npy`: mode diagnostics.
- `nonlinear_pgd_modes.npz`: separated modes `Y`, `G`, and `time`.
- `nonlinear_pgd_flux_step_50_0p005s.png`: `|B|` at `t=0.005 s`.
- `nonlinear_pgd_flux_step_100_0p010s.png`: `|B|` at `t=0.01 s`.
- `nonlinear_pgd_flux_node_5209.png`: `|B|` history at node `5209`.
