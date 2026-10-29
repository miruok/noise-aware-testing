# Copyright (c) 2026 Mir Mohammad Yousuf and Shabir Ahmad Sofi,
# National Institute of Technology Srinagar. Released under the MIT License.
"""Paper figures (PDF + PNG) from the JSON results.

Categorical colours follow a fixed order (blue, orange, aqua, yellow, magenta);
every series also carries a distinct marker or hatch so identity never depends
on colour alone. Output: results/figures/.
"""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import RESULTS as RES, PRETTY
from matplotlib.lines import Line2D
from nisq_framework.stats import wilson

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MK = ["o", "s", "^", "D", "v"]
HATCH = ["", "//", "..", "\\\\", "xx"]
OUT = os.path.join(RES, "figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "legend.fontsize": 8, "axes.titlesize": 9.5, "mathtext.default": "regular", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#888888", "legend.frameon": False, "lines.linewidth": 2})


def load(*a):
    p = os.path.join(RES, *a)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


def fig_soundness():
    e1 = load("rq1_soundness", "summary.json")
    if not e1: return
    order = ["nao_a0.001", "nao1_a0.001", "nao_a0.01", "nao1_a0.01", "spc_envelope",
             "qoin_style", "fixed_0.1", "fixed_0.05", "fixed_0.02", "fixed_0.01", "fixed_0.005"]
    labels = [r"NAO, $\alpha=10^{-3}$", r"NAO-1, $\alpha=10^{-3}$", r"NAO, $\alpha=10^{-2}$", r"NAO-1, $\alpha=10^{-2}$", "SPC envelope",
              "QOIN-style filter", r"fixed $\tau=0.10$", r"fixed $\tau=0.05$", r"fixed $\tau=0.02$", r"fixed $\tau=0.01$", r"fixed $\tau=0.005$"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    y = np.arange(len(order))[::-1]
    for yi, o in zip(y, order):
        v = e1["by_oracle"][o]; est = max(v["fpr"], 1e-5)
        lo, hi = max(v["fpr_cluster_ci"][0], 1e-5), max(v["fpr_cluster_ci"][1], 1e-5)
        col = C[0] if o.startswith("nao_") else C[1] if o.startswith("nao1") else C[2] if o == "spc_envelope" else C[3] if o == "qoin_style" else "#777777"
        ax.plot([lo, hi], [yi, yi], color=col, lw=2); ax.plot(est, yi, MK[0], color=col, ms=6)
        if v["fpr_cluster_ci"][0] <= 0:
            ax.plot(1e-5, yi, "<", color=col, ms=6)
    for a, ls in ((1e-3, "--"), (1e-2, ":")):
        ax.axvline(a, color="#555555", ls=ls, lw=1)
        ax.text(a * 1.12, y.max() + 0.7, rf"$\alpha={a:g}$", ha="left", fontsize=8, color="#555555")
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("false-positive rate on correct runs (log scale)")
    ax.set_xlim(7e-6, 1.5)
    ax.set_ylim(-0.7, y.max() + 1.2)
    save(fig, "fig_soundness_fpr_vs_alpha")


def fig_power():
    e2 = load("rq4_faults", "summary.json"); pa = load("rq4_faults", "power_validation.json")
    if not e2: return
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.2))
    ax = axs[0]
    if pa:
        for i, dd in enumerate(sorted({r["d"] for r in pa})):
            R = sorted([r for r in pa if r["d"] == dd], key=lambda r: r["delta"])
            ax.plot([r["delta"] for r in R], [r["predicted"] for r in R], color=C[i], lw=1.5, label=f"d = {dd}")
            ax.plot([r["delta"] for r in R], [r["empirical"] for r in R], MK[i], color=C[i], ms=4, ls="none")
        ax.set_xscale("log"); ax.set_xlabel("true Jensen-Shannon divergence (log scale)")
        ax.set_ylabel("detection probability"); ax.set_title("(a) predicted (lines) vs simulated (markers)")
        ax.set_ylim(-0.03, 1.03); ax.legend(title="outcomes", title_fontsize=8)
    ax = axs[1]
    for i, mode in enumerate(("nao", "nao1")):
        cal = [c for c in e2["power_calibration"] if c["oracle"] == mode]
        ax.plot([c["mean_predicted"] for c in cal], [c["observed"] for c in cal], MK[i] + "-", color=C[i], ms=5,
                label={"nao": "NAO", "nao1": "NAO-1"}[mode])
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="ideal (y = x)")
    ax.set_xlabel(r"predicted recall (noncentral $\chi^2$)"); ax.set_ylabel("observed recall")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect("equal", adjustable="box")
    ax.set_title("(b) injected faults, binned by prediction"); ax.legend(loc="upper left")
    save(fig, "fig_power")
    # recall by noise
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    noises = list(next(iter(e2["by_noise"].values())).keys())
    for i, o in enumerate(["nao", "nao1", "spc_envelope", "fixed_0.05"]):
        ax.plot(range(len(noises)), [e2["by_noise"][o][n]["recall"] for n in noises], MK[i] + "-", color=C[i], ms=5,
                label={"nao": "NAO", "nao1": "NAO-1", "spc_envelope": "SPC envelope", "fixed_0.05": r"fixed $\tau=0.05$ (ideal reference)"}[o])
    ax.set_xticks(range(len(noises))); ax.set_xticklabels([n.replace("_", " ") for n in noises])
    ax.set_xlabel("synthetic noise level")
    ax.set_ylabel("recall on non-equivalent faults"); ax.set_ylim(0, 1.02); ax.legend(loc="lower left")
    save(fig, "fig_recall_by_noise")


def fig_mutation():
    e3 = load("rq2_mutation", "summary.json")
    if not e3: return
    cls = ["semantic", "noise_amplifying", "noise_neutral"]
    names = ["semantic\n(mutation score)", "noise-amplifying\n(equivalent)", "noise-neutral\n(equivalent)"]
    fig, ax = plt.subplots(figsize=(4.8, 3.2)); w = 0.26
    for i, o in enumerate(["nao", "nao1", "spc_envelope"]):
        R = e3["rejection_by_class"][o]
        vals = [R[c]["rate"] for c in cls]
        ci = [wilson(R[c]["k"], R[c]["n"]) for c in cls]
        err = np.array([[v - lo for v, (lo, hi) in zip(vals, ci)], [hi - v for v, (lo, hi) in zip(vals, ci)]])
        xs = np.arange(3) + (i - 1) * w
        ax.bar(xs, vals, w - 0.02, color=C[i], hatch=HATCH[i], edgecolor="white",
               label={"nao": "NAO", "nao1": "NAO-1", "spc_envelope": "SPC envelope"}[o])
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor="#222222", elinewidth=1, capsize=3)
        for x_, v, (lo, hi) in zip(xs, vals, ci):
            ax.text(x_, hi + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(names); ax.set_ylabel("fraction of mutants rejected"); ax.set_ylim(0, 1.12)
    ax.legend(ncol=3, loc="upper right")
    save(fig, "fig_mutation_by_class")


def fig_drift():
    e5 = load("rq5_drift_cost", "results.json")
    if not e5: return
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.6))
    SIZE = [8, 6.5, 5, 4]
    benches = sorted({x["benchmark"] for x in e5["between_window"]})
    for i, b in enumerate(benches):
        R = [x for x in e5["between_window"] if x["benchmark"] == b]
        axs[0].plot([x["r"] for x in R], [x["fpr"]["nao"] for x in R], MK[i] + "-", color=C[i], ms=SIZE[i], label=PRETTY[b], zorder=3 + i)
        axs[0].plot([x["r"] for x in R], [x["fpr_with_sentinel"]["nao"] for x in R], MK[i] + ":", color=C[i], ms=SIZE[i], lw=1.6, zorder=3 + i)
        W = [x for x in e5["within_window"] if x["benchmark"] == b]
        axs[1].plot([x["r"] for x in W], [x["detection_rate"] for x in W], MK[i] + "-", color=C[i], ms=SIZE[i], label=PRETTY[b], zorder=3 + i)
    axs[0].set_xlabel(r"relative noise increase after calibration, $r$"); axs[0].set_ylabel("false-positive rate of NAO")
    axs[0].set_title("(a) drift between calibration and testing"); axs[0].set_ylim(-0.03, 1.05)
    h_b = axs[0].get_legend_handles_labels()
    h_b = ([Line2D([], [], color=C[i], marker=MK[i], ms=5, lw=1.8) for i in range(len(benches))], h_b[1])
    style = [Line2D([], [], color="#444444", ls="-", lw=1.8), Line2D([], [], color="#444444", ls=":", lw=1.8)]
    fig.legend(h_b[0] + style, h_b[1] + ["(a) no recalibration", r"(a) one-run sentinel, $\alpha_s=10^{-3}$"],
               loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    axs[1].set_xlabel(r"noise drift within the calibration window, $r$"); axs[1].set_ylabel("drift-detection rate")
    axs[1].set_title(r"(b) homogeneity test on $K = 8$ runs"); axs[1].set_ylim(-0.03, 1.05)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_drift.{ext}"), dpi=200)
    plt.close(fig)


def fig_scaling():
    e7 = load("rq4_scaling", "results.json")
    if not e7: return
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1))
    x = np.arange(len(e7)); labs = [f"{r['circuit'].split('_')[0].upper()}-{r['qubits']}" for r in e7]
    series = (("nao", "NAO (full distribution)"), ("nao_marginal", "NAO (2-qubit marginals)"), ("spc_envelope", "SPC envelope"))
    for ax, key, ylab in ((axs[0], "fpr", "false-positive rate"), (axs[1], "recall", "recall on non-equivalent faults")):
        for i, (k, lab) in enumerate(series):
            ax.plot(x + (i - 1) * 0.18, [r[key][k] for r in e7], MK[i], color=C[i], ms=6, ls="none", label=lab)
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8); ax.set_ylabel(ylab)
        ax.axvline(4.5, color="#bbbbbb", lw=1)
    axs[0].axhline(1e-3, color="#555555", ls="--", lw=1); axs[0].text(0, 1e-3 + 0.002, "α = 1e-3", fontsize=7, color="#555555")
    axs[0].set_ylim(-0.005, 0.08); axs[1].set_ylim(0.6, 1.03)
    axs[0].set_title("(a) correct runs, η = 0.005, 4096 shots", fontsize=9); axs[1].set_title("(b) injected faults", fontsize=9)
    axs[1].legend(fontsize=7, loc="lower left")
    save(fig, "fig_scaling")


if __name__ == "__main__":
    for f in (fig_soundness, fig_power, fig_mutation, fig_drift, fig_scaling):
        f()
    print(sorted(os.listdir(OUT)))
