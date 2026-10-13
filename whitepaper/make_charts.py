"""Charts for the ecfr-mcp retrieval whitepaper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/home/hatch/workspace/ecfr-mcp/whitepaper/charts")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "figure.dpi": 150, "savefig.bbox": "tight",
})

systems = ["BM25", "RRF\nhybrid", "bge-rerank\n(base)", "Vectors\n+ citation",
           "Chunked", "HyDE", "HyDE\n+ query", "LLM rerank", "HyDE+\nrerank"]
r1 = np.array([0.28, 0.50, 0.52, 0.64, 0.62, 0.68, 0.72, 0.76, 0.74])
r3 = np.array([0.44, 0.74, 0.70, 0.78, 0.80, 0.84, 0.82, 0.94, 0.90])
r5 = np.array([0.54, 0.80, 0.74, 0.84, 0.86, 0.90, 0.88, 0.96, 0.94])
colors = {"r@1": "#1a56db", "r@3": "#3b82f6", "r@5": "#93c5fd"}

# 1. grouped bars, all systems
x = np.arange(len(systems)); w = 0.26
fig, ax = plt.subplots(figsize=(11, 5))
for i, (vals, lab) in enumerate([(r1, "r@1"), (r3, "r@3"), (r5, "r@5")]):
    bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=colors[lab])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(systems)
ax.set_ylabel("Recall"); ax.set_ylim(0, 1.08)
ax.set_title("Recall across all retrieval configurations (n=50 frozen queries)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.savefig(OUT / "recall_all.png"); plt.close(fig)

# 2. per-type r@1 for the key systems
labels = ["Vectors\n+ citation", "HyDE\n+ query", "LLM rerank"]
nat = [0.667, 0.758, 0.788]
par = [0.417, 0.500, 0.583]
x = np.arange(len(labels)); w = 0.34
fig, ax = plt.subplots(figsize=(8, 4.6))
b1 = ax.bar(x - w / 2, nat, w, label="natural (n=33)", color="#1a56db")
b2 = ax.bar(x + w / 2, par, w, label="paraphrase (n=12)", color="#f59e0b")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Recall@1"); ax.set_ylim(0, 1.0)
ax.set_title("Where the gains came from: recall@1 by query type")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.savefig(OUT / "recall_by_type.png"); plt.close(fig)

# 3. cost vs recall frontier
names = ["BM25", "RRF hybrid", "bge-rerank", "Vectors+cite", "Chunked",
         "HyDE", "HyDE+query", "LLM rerank", "HyDE+rerank"]
cost = np.array([0, 0, 0, 0, 0, 0.0056, 0.0056, 0.0861, 0.18])
rr1 = np.array([0.28, 0.50, 0.52, 0.64, 0.62, 0.68, 0.72, 0.76, 0.74])
fig, ax = plt.subplots(figsize=(8.5, 5))
free = cost == 0
ax.scatter(np.zeros(free.sum()) - 0.35, rr1[free], s=90, color="#6b7280", zorder=3)
for i in np.where(free)[0]:
    ax.text(-0.5, rr1[i] + 0.012, names[i], fontsize=8, ha="center", color="#374151")
paid = ~free
ax.scatter(cost[paid], rr1[paid], s=110, color="#1a56db", zorder=3)
for i in np.where(paid)[0]:
    ax.annotate(names[i], (cost[i], rr1[i]), xytext=(8, 6),
                textcoords="offset points", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d1d5db", alpha=0.9))
ax.set_xscale("symlog", linthresh=0.001)
ax.set_xlabel("Experiment cost per 50-query eval (USD, log scale; $0 shown at left)")
ax.set_ylabel("Recall@1")
ax.set_title("The efficiency frontier: recall@1 vs. experiment cost")
ax.set_xlim(-0.9, 0.5); ax.grid(alpha=0.3)
fig.savefig(OUT / "cost_frontier.png"); plt.close(fig)

# 4. r@1 progression across rounds
rounds = ["Round 2\nvectors", "Round 3\n+cite fast-path", "Round 4\nchunk/rerank\nattempts",
          "Round 5\nHyDE", "Round 6\nLLM rerank", "Round 7\ncombo"]
prog = [0.54, 0.64, 0.64, 0.72, 0.76, 0.74]
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(rounds, prog, marker="o", ms=8, color="#1a56db", lw=2.5)
ax.fill_between(rounds, prog, alpha=0.12, color="#1a56db")
for i, v in enumerate(prog):
    ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=10, weight="bold")
ax.set_ylabel("Recall@1 (best config that round)")
ax.set_ylim(0.45, 0.85)
ax.set_title("Recall@1 progression: seven rounds on the frozen 50-query set")
ax.grid(alpha=0.3)
fig.savefig(OUT / "progression.png"); plt.close(fig)

print("charts written:", sorted(p.name for p in OUT.glob("*.png")))
