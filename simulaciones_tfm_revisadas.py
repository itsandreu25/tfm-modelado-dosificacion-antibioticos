from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, brentq


@dataclass(frozen=True)
class ModelParameters:
    r_s: float = 1.20
    r_r: float = 0.95
    carrying_capacity: float = 1.0e9
    mutation_rate: float = 1.0e-8
    elimination_rate: float = 3.0
    emax_s: float = 3.8
    emax_r: float = 1.2
    ec50_s: float = 1.0
    ec50_r: float = 5.0
    hill_s: float = 2.0
    hill_r: float = 2.0


@dataclass(frozen=True)
class Regimen:
    rate: float
    interval: float
    duration: float


DEFAULT_PARAMETERS = ModelParameters()
INITIAL_STATE = np.array([1.0e6, 1.0e3, 0.0], dtype=float)
BASE_HORIZON = 7.0
Q_MAX = 12.0
WEIGHTS = (0.70, 0.20, 0.10)


def hill_effect(concentration, emax, ec50, exponent):
    concentration = max(float(concentration), 0.0)
    if concentration == 0.0:
        return 0.0
    return emax * concentration**exponent / (
        ec50**exponent + concentration**exponent
    )


def dose_input(time, regimen):
    if regimen.rate <= 0.0:
        return 0.0
    return regimen.rate if (time % regimen.interval) < regimen.duration else 0.0


def model_rhs(time, state, regimen, parameters):
    susceptible, resistant, concentration = np.maximum(state, 0.0)
    total = susceptible + resistant
    competition = 1.0 - total / parameters.carrying_capacity

    effect_s = hill_effect(
        concentration, parameters.emax_s, parameters.ec50_s, parameters.hill_s
    )
    effect_r = hill_effect(
        concentration, parameters.emax_r, parameters.ec50_r, parameters.hill_r
    )

    d_s = (
        parameters.r_s * susceptible * competition
        - effect_s * susceptible
        - parameters.mutation_rate * susceptible
    )
    d_r = (
        parameters.r_r * resistant * competition
        - effect_r * resistant
        + parameters.mutation_rate * susceptible
    )
    d_a = -parameters.elimination_rate * concentration + dose_input(time, regimen)
    return np.array([d_s, d_r, d_a], dtype=float)


def simulate(
    regimen,
    horizon=BASE_HORIZON,
    parameters=DEFAULT_PARAMETERS,
    initial_state=INITIAL_STATE,
    points_per_day=100,
    rtol=1.0e-7,
    atol=(1.0e-2, 1.0e-5, 1.0e-9),
    max_step=0.03,
):
    number_of_points = max(int(horizon * points_per_day) + 1, 401)
    evaluation_times = np.linspace(0.0, horizon, number_of_points)

    solution = solve_ivp(
        fun=lambda t, y: model_rhs(t, y, regimen, parameters),
        t_span=(0.0, horizon),
        y0=np.asarray(initial_state, dtype=float),
        method="LSODA",
        t_eval=evaluation_times,
        rtol=rtol,
        atol=np.asarray(atol, dtype=float),
        max_step=max_step,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


def exact_total_dose(regimen, horizon):
    if regimen.rate <= 0.0:
        return 0.0
    starts = np.arange(0.0, horizon, regimen.interval)
    durations = np.minimum(regimen.duration, np.maximum(horizon - starts, 0.0))
    return float(regimen.rate * np.sum(durations))


def calculate_metrics(solution, regimen, parameters=DEFAULT_PARAMETERS, weights=WEIGHTS):
    times = solution.t
    susceptible, resistant, concentration = solution.y
    horizon = float(times[-1])

    auc_resistant = float(np.trapezoid(resistant, times))
    auc_total = float(np.trapezoid(susceptible + resistant, times))
    dose = exact_total_dose(regimen, horizon)
    final_total = float(susceptible[-1] + resistant[-1])
    final_fraction = float(resistant[-1] / (final_total + 1.0e-30))

    component_total = weights[0] * auc_total / (
        parameters.carrying_capacity * horizon
    )
    component_resistant = weights[1] * auc_resistant / (
        parameters.carrying_capacity * horizon
    )
    component_dose = weights[2] * dose / (Q_MAX * horizon)

    return {
        "q": regimen.rate,
        "tau": regimen.interval,
        "delta": regimen.duration,
        "AUC_R": auc_resistant,
        "AUC_N": auc_total,
        "Dose": dose,
        "rho_T": final_fraction,
        "S_T": float(susceptible[-1]),
        "R_T": float(resistant[-1]),
        "N_T": final_total,
        "A_max": float(np.max(concentration)),
        "J_N": component_total,
        "J_R": component_resistant,
        "J_D": component_dose,
        "J": component_total + component_resistant + component_dose,
    }


def objective_vector(vector, parameters=DEFAULT_PARAMETERS):
    regimen = Regimen(float(vector[0]), float(vector[1]), float(vector[2]))
    try:
        solution = simulate(
            regimen,
            parameters=parameters,
            points_per_day=45,
            rtol=3.0e-7,
            atol=(1.0e-1, 1.0e-4, 1.0e-8),
            max_step=0.05,
        )
        return calculate_metrics(solution, regimen, parameters)["J"]
    except Exception:
        return 1.0e6


def optimize_regimen(seed=2026):
    result = differential_evolution(
        objective_vector,
        bounds=[(0.0, 12.0), (0.5, 2.0), (0.05, 0.5)],
        seed=seed,
        maxiter=35,
        popsize=8,
        tol=1.0e-5,
        polish=True,
        updating="immediate",
        workers=1,
    )
    return Regimen(*map(float, result.x)), result


def nondominated_mask(values):
    count = values.shape[0]
    mask = np.ones(count, dtype=bool)
    for i in range(count):
        if not mask[i]:
            continue
        better = np.all(values <= values[i], axis=1) & np.any(
            values < values[i], axis=1
        )
        if np.any(better):
            mask[i] = False
    return mask


def equilibrium_branches_constant_input(q_values, parameters=DEFAULT_PARAMETERS):
    rows = []
    for q in q_values:
        concentration = q / parameters.elimination_rate
        effect_s = hill_effect(
            concentration, parameters.emax_s, parameters.ec50_s, parameters.hill_s
        )
        effect_r = hill_effect(
            concentration, parameters.emax_r, parameters.ec50_r, parameters.hill_r
        )
        invasion_s = parameters.r_s - effect_s
        invasion_r = parameters.r_r - effect_r

        susceptible_eq = 0.0
        resistant_eq = 0.0
        if max(invasion_s, invasion_r) > 0.0:
            if invasion_s >= invasion_r:
                susceptible_eq = parameters.carrying_capacity * max(
                    1.0 - effect_s / parameters.r_s, 0.0
                )
            else:
                resistant_eq = parameters.carrying_capacity * max(
                    1.0 - effect_r / parameters.r_r, 0.0
                )

        rows.append(
            {
                "q": q,
                "A_star": concentration,
                "g_S": invasion_s,
                "g_R": invasion_r,
                "S_eq": susceptible_eq,
                "R_eq": resistant_eq,
            }
        )
    return pd.DataFrame(rows)


def analytical_thresholds(parameters=DEFAULT_PARAMETERS):
    def effect_s(a):
        return hill_effect(a, parameters.emax_s, parameters.ec50_s, parameters.hill_s)

    def effect_r(a):
        return hill_effect(a, parameters.emax_r, parameters.ec50_r, parameters.hill_r)

    selection_concentration = brentq(
        lambda a: (parameters.r_s - effect_s(a))
        - (parameters.r_r - effect_r(a)),
        0.0,
        20.0,
    )
    susceptible_zero = parameters.ec50_s * (
        parameters.r_s / (parameters.emax_s - parameters.r_s)
    ) ** (1.0 / parameters.hill_s)
    resistant_zero = parameters.ec50_r * (
        parameters.r_r / (parameters.emax_r - parameters.r_r)
    ) ** (1.0 / parameters.hill_r)

    return {
        "q_selection": parameters.elimination_rate * selection_concentration,
        "q_susceptible_zero": parameters.elimination_rate * susceptible_zero,
        "q_resistant_zero": parameters.elimination_rate * resistant_zero,
    }


def run_all(output_directory="."):
    output_directory = Path(output_directory)
    figures_directory = output_directory / "figuras"
    data_directory = output_directory / "datos"
    figures_directory.mkdir(parents=True, exist_ok=True)
    data_directory.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    optimized_regimen, optimizer_result = optimize_regimen(seed=2026)

    regimens = {
        "Sin tratamiento": Regimen(0.0, 1.0, 0.1),
        "Baja intensidad": Regimen(2.0, 1.0, 0.1),
        "Alta intensidad": Regimen(8.0, 1.0, 0.1),
        "Optimizada": optimized_regimen,
    }

    rows = []
    trajectories = {}
    for name, regimen in regimens.items():
        solution = simulate(regimen, points_per_day=300, max_step=0.015)
        row = calculate_metrics(solution, regimen)
        row["Escenario"] = name
        rows.append(row)
        trajectories[name] = solution

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(data_directory / "resultados_escenarios.csv", index=False)

    trajectory_rows = []
    for name, solution in trajectories.items():
        for t, s, r, a in zip(solution.t, *solution.y):
            trajectory_rows.append(
                {"Escenario": name, "t": t, "S": s, "R": r, "A": a}
            )
    pd.DataFrame(trajectory_rows).to_csv(
        data_directory / "trayectorias_escenarios.csv", index=False
    )

    plt.figure(figsize=(8.6, 5.2))
    for name, solution in trajectories.items():
        if name != "Sin tratamiento":
            plt.plot(solution.t, solution.y[2], label=name)
    plt.xlabel("Tiempo (días)")
    plt.ylabel("Concentración relativa A(t)")
    plt.title("Perfiles de concentración para las pautas comparadas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_directory / "fig_01_concentracion_escenarios.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8.6, 5.2))
    for name, solution in trajectories.items():
        plt.plot(solution.t, solution.y[0], label=name)
    plt.yscale("log")
    plt.xlabel("Tiempo (días)")
    plt.ylabel("Bacterias sensibles S(t) [UFC/mL]")
    plt.title("Evolución de la población sensible")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_directory / "fig_02_sensibles_escenarios.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8.6, 5.2))
    for name, solution in trajectories.items():
        plt.plot(solution.t, solution.y[1], label=name)
    plt.yscale("log")
    plt.xlabel("Tiempo (días)")
    plt.ylabel("Bacterias resistentes R(t) [UFC/mL]")
    plt.title("Evolución de la población resistente")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_directory / "fig_03_resistentes_escenarios.png", dpi=220)
    plt.close()

    ordered = metrics_frame.set_index("Escenario").loc[list(regimens)]
    x = np.arange(len(ordered))
    width = 0.25
    plt.figure(figsize=(8.8, 5.2))
    plt.bar(x - width, ordered["J_N"], width, label="Carga total")
    plt.bar(x, ordered["J_R"], width, label="Carga resistente")
    plt.bar(x + width, ordered["J_D"], width, label="Dosis")
    plt.xticks(x, ordered.index, rotation=12)
    plt.ylabel("Contribución ponderada")
    plt.title("Componentes de la función objetivo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_directory / "fig_04_componentes_objetivo.png", dpi=220)
    plt.close()

    q_rows = []
    for q in np.linspace(0.0, 12.0, 49):
        regimen = Regimen(float(q), 1.0, 0.1)
        solution = simulate(
            regimen,
            points_per_day=55,
            rtol=5.0e-7,
            atol=(1.0e-1, 1.0e-4, 1.0e-8),
            max_step=0.05,
        )
        q_rows.append(calculate_metrics(solution, regimen))
    q_frame = pd.DataFrame(q_rows)
    q_frame.to_csv(data_directory / "barrido_q_periodico.csv", index=False)

    plt.figure(figsize=(8.6, 5.2))
    plt.plot(q_frame["q"], q_frame["AUC_R"])
    plt.xlabel("Tasa de infusión q")
    plt.ylabel("Carga resistente acumulada")
    plt.title(r"Barrido sistemático de q con tau=1 y delta=0.1")
    plt.tight_layout()
    plt.savefig(figures_directory / "fig_05_barrido_q_auc_resistente.png", dpi=220)
    plt.close()

    q_grid = np.linspace(0.0, 12.0, 25)
    tau_grid = np.linspace(0.5, 2.0, 16)
    grid_rows = []
    for tau in tau_grid:
        for q in q_grid:
            regimen = Regimen(float(q), float(tau), 0.1)
            solution = simulate(
                regimen,
                points_per_day=40,
                rtol=8.0e-7,
                atol=(1.0, 1.0e-3, 1.0e-8),
                max_step=0.06,
            )
            grid_rows.append(calculate_metrics(solution, regimen))
    grid_frame = pd.DataFrame(grid_rows)
    grid_frame.to_csv(data_directory / "barrido_q_tau.csv", index=False)

    matrix_r = grid_frame.pivot(index="tau", columns="q", values="AUC_R")
    plt.figure(figsize=(8.6, 5.4))
    image = plt.imshow(
        matrix_r.values,
        origin="lower",
        aspect="auto",
        extent=[q_grid.min(), q_grid.max(), tau_grid.min(), tau_grid.max()],
    )
    plt.xlabel("Tasa de infusión q")
    plt.ylabel("Intervalo tau (días)")
    plt.title("Carga resistente acumulada en la malla (q,tau)")
    plt.colorbar(image, label="AUC_R")
    plt.tight_layout()
    plt.savefig(figures_directory / "fig_06_mapa_q_tau_auc_resistente.png", dpi=220)
    plt.close()

    pareto_rows = []
    for q in np.linspace(0.0, 12.0, 17):
        for tau in np.linspace(0.5, 2.0, 9):
            for delta in np.linspace(0.05, 0.5, 6):
                regimen = Regimen(float(q), float(tau), float(delta))
                solution = simulate(
                    regimen,
                    points_per_day=32,
                    rtol=1.0e-6,
                    atol=(1.0, 1.0e-3, 1.0e-8),
                    max_step=0.07,
                )
                pareto_rows.append(calculate_metrics(solution, regimen))
    pareto_frame = pd.DataFrame(pareto_rows)
    mask = nondominated_mask(
        pareto_frame[["AUC_N", "AUC_R", "Dose"]].to_numpy(dtype=float)
    )
    pareto_frame["No_dominada"] = mask
    pareto_frame.to_csv(data_directory / "exploracion_pareto_grid.csv", index=False)

    plt.figure(figsize=(8.6, 5.4))
    dominated = pareto_frame[~pareto_frame["No_dominada"]]
    nondominated = pareto_frame[pareto_frame["No_dominada"]]
    plt.scatter(
        dominated["Dose"], dominated["AUC_R"], s=13, alpha=0.45,
        label="Pautas dominadas"
    )
    plt.scatter(
        nondominated["Dose"], nondominated["AUC_R"], s=34, marker="x",
        label="Pautas no dominadas"
    )
    plt.xlabel("Dosis relativa total")
    plt.ylabel("Carga resistente acumulada")
    plt.title("Exploración sistemática dosis-resistencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_directory / "fig_07_pareto_dosis_resistencia.png", dpi=220)
    plt.close()

    q_long = np.linspace(0.0, 40.0, 401)
    equilibrium_frame = equilibrium_branches_constant_input(q_long)
    equilibrium_frame.to_csv(
        data_directory / "equilibrios_exposicion_constante.csv", index=False
    )
    thresholds = analytical_thresholds()
    pd.DataFrame([thresholds]).to_csv(
        data_directory / "umbrales_analiticos.csv", index=False
    )

    plt.figure(figsize=(8.8, 5.4))
    plt.plot(
        equilibrium_frame["q"],
        equilibrium_frame["S_eq"] / DEFAULT_PARAMETERS.carrying_capacity,
        label="S*/K",
    )
    plt.plot(
        equilibrium_frame["q"],
        equilibrium_frame["R_eq"] / DEFAULT_PARAMETERS.carrying_capacity,
        label="R*/K",
    )
    plt.axvline(
        thresholds["q_selection"], linestyle="--",
        label="Cambio de ventaja q_sel"
    )
    plt.axvline(
        thresholds["q_resistant_zero"], linestyle=":",
        label="Umbral resistente q_R"
    )
    plt.xlabel("Entrada constante q")
    plt.ylabel("Equilibrio normalizado")
    plt.title("Respuesta estacionaria aproximada bajo exposición constante")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_directory / "fig_08_equilibrio_exposicion_constante.png", dpi=220
    )
    plt.close()

    baseline_solution = simulate(
        optimized_regimen, points_per_day=180, max_step=0.02
    )
    baseline_metrics = calculate_metrics(
        baseline_solution, optimized_regimen, DEFAULT_PARAMETERS
    )

    fields = [
        "r_s", "r_r", "mutation_rate", "elimination_rate",
        "emax_s", "emax_r", "ec50_s", "ec50_r"
    ]
    sensitivity_rows = []
    change = 0.20
    for field in fields:
        nominal = getattr(DEFAULT_PARAMETERS, field)
        lower_parameters = replace(
            DEFAULT_PARAMETERS, **{field: nominal * (1.0 - change)}
        )
        upper_parameters = replace(
            DEFAULT_PARAMETERS, **{field: nominal * (1.0 + change)}
        )

        lower_solution = simulate(
            optimized_regimen,
            parameters=lower_parameters,
            points_per_day=70,
            rtol=6.0e-7,
            atol=(1.0e-1, 1.0e-4, 1.0e-8),
            max_step=0.05,
        )
        upper_solution = simulate(
            optimized_regimen,
            parameters=upper_parameters,
            points_per_day=70,
            rtol=6.0e-7,
            atol=(1.0e-1, 1.0e-4, 1.0e-8),
            max_step=0.05,
        )
        lower_metrics = calculate_metrics(
            lower_solution, optimized_regimen, lower_parameters
        )
        upper_metrics = calculate_metrics(
            upper_solution, optimized_regimen, upper_parameters
        )

        sensitivity_rows.append({
            "Parametro": field,
            "Sens_AUC_R": (
                (upper_metrics["AUC_R"] - lower_metrics["AUC_R"])
                / (2.0 * change * baseline_metrics["AUC_R"])
            ),
            "Sens_AUC_N": (
                (upper_metrics["AUC_N"] - lower_metrics["AUC_N"])
                / (2.0 * change * baseline_metrics["AUC_N"])
            ),
            "AUC_R_menos20": lower_metrics["AUC_R"],
            "AUC_R_mas20": upper_metrics["AUC_R"],
            "AUC_N_menos20": lower_metrics["AUC_N"],
            "AUC_N_mas20": upper_metrics["AUC_N"],
        })

    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    sensitivity_frame.to_csv(data_directory / "sensibilidad_local.csv", index=False)

    ordered_sensitivity = sensitivity_frame.reindex(
        sensitivity_frame["Sens_AUC_R"].abs().sort_values().index
    )
    plt.figure(figsize=(8.8, 5.4))
    plt.barh(
        ordered_sensitivity["Parametro"],
        ordered_sensitivity["Sens_AUC_R"],
    )
    plt.xlabel("Sensibilidad relativa de AUC_R")
    plt.ylabel("Parámetro")
    plt.title("Sensibilidad local alrededor de la pauta optimizada")
    plt.tight_layout()
    plt.savefig(figures_directory / "fig_09_sensibilidad_local.png", dpi=220)
    plt.close()

    execution_summary = {
        "optimized_q": optimized_regimen.rate,
        "optimized_tau": optimized_regimen.interval,
        "optimized_delta": optimized_regimen.duration,
        "optimizer_fun": float(optimizer_result.fun),
        "optimizer_nfev": int(optimizer_result.nfev),
        "optimizer_success": bool(optimizer_result.success),
        "optimizer_message": str(optimizer_result.message),
        "execution_seconds": time.perf_counter() - start,
        **thresholds,
    }
    pd.DataFrame([execution_summary]).to_csv(
        data_directory / "resumen_ejecucion.csv", index=False
    )
    return {
        "metrics": metrics_frame,
        "summary": execution_summary,
        "thresholds": thresholds,
    }


if __name__ == "__main__":
    result = run_all(Path(__file__).resolve().parent)
    print(result["metrics"].to_string(index=False))
    print(result["summary"])
