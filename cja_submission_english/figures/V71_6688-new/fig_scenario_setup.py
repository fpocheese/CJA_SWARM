#!/usr/bin/env python3
"""Scenario setup schematic: top-down view with randomization ranges."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Arc
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(1, 1, figsize=(5.8, 3.4))

# ── Deployment regions ──
# Attackers: x∈[-1400,-950], y∈[-200,250]
att_rect = Rectangle((-1400, -200), 450, 450, linewidth=1.8,
                     edgecolor='#d62728', facecolor='#d6272810',
                     linestyle='--', zorder=2)
ax.add_patch(att_rect)

# Defenders: x∈[370,900], y∈[-260,260]
def_rect = Rectangle((370, -260), 530, 520, linewidth=1.8,
                     edgecolor='#1f77b4', facecolor='#1f77b410',
                     linestyle='--', zorder=2)
ax.add_patch(def_rect)

# ── HVT ──
ax.plot(1200, 0, marker='*', markersize=16, color='#2ca02c',
        markeredgecolor='k', markeredgewidth=0.6, zorder=5)
ax.text(1200, -50, r'$p_H$', ha='center', fontsize=10, color='#2ca02c',
        fontweight='bold')

# ── Sample positions with heading arrows (4v4 case) ──
att_pos = [(-1091, 91), (-1182, 178), (-1254, 72), (-1293, -102)]
att_psi = [-3.1, -10.1, 6.3, 7.6]
for px, py in att_pos:
    ax.plot(px, py, 'o', color='#d62728', markersize=4.5, zorder=4)

def_pos = [(733, 35), (636, 149), (379, 41), (644, -132)]
def_psi = [172.9, 170.1, -170.9, 174.4]
for px, py in def_pos:
    ax.plot(px, py, 's', color='#1f77b4', markersize=4.5, zorder=4)

# ── Heading range arcs ──
# Attacker heading: psi_A ∈ [-11°, +11°] around 0°
# Show a representative arrow + arc
cx_a, cy_a = -950, 50
L = 90
ax.annotate('', xy=(cx_a+L, cy_a), xytext=(cx_a, cy_a),
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.3))
# arc showing ±11°
arc_a = Arc((cx_a, cy_a), 120, 120, angle=0, theta1=-11, theta2=11,
            color='#d62728', lw=1.0, linestyle='-')
ax.add_patch(arc_a)
ax.text(cx_a+95, cy_a+20, r'$\pm 11^\circ$', fontsize=7.5, color='#d62728')

# Defender heading: psi_D ∈ [160°, 180°]≈180°±15°
cx_d, cy_d = 900, -50
ax.annotate('', xy=(cx_d-L, cy_d), xytext=(cx_d, cy_d),
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.3))
arc_d = Arc((cx_d, cy_d), 120, 120, angle=180, theta1=-15, theta2=15,
            color='#1f77b4', lw=1.0, linestyle='-')
ax.add_patch(arc_d)
ax.text(cx_d-155, cy_d+22, r'$\pm 15^\circ$', fontsize=7.5, color='#1f77b4')

# ── Dimension labels ──
# Attacker x-range
ax.annotate('', xy=(-950, -225), xytext=(-1400, -225),
            arrowprops=dict(arrowstyle='<->', color='0.3', lw=0.7))
ax.text(-1175, -250, r'$\Delta x_A\!\approx\!450$\,m', ha='center', fontsize=7.5, color='0.3')

# Attacker y-range
ax.annotate('', xy=(-1425, 250), xytext=(-1425, -200),
            arrowprops=dict(arrowstyle='<->', color='0.3', lw=0.7))
ax.text(-1520, 25, r'$\Delta y_A$' + '\n' + r'$\approx\!450$m',
        ha='center', fontsize=7, color='0.3')

# Defender x-range
ax.annotate('', xy=(900, -285), xytext=(370, -285),
            arrowprops=dict(arrowstyle='<->', color='0.3', lw=0.7))
ax.text(635, -310, r'$\Delta x_D\!\approx\!530$\,m', ha='center', fontsize=7.5, color='0.3')

# Overall distance to HVT
ax.annotate('', xy=(1200, -345), xytext=(-1175, -345),
            arrowprops=dict(arrowstyle='<->', color='0.45', lw=0.85,
                            linestyle='dashed'))
ax.text(12, -372, r'$\rho_{AH}(0)\!\approx\!2200$--$2500$\,m',
        ha='center', fontsize=8, color='0.45')

# ── Text annotations ──
# Region labels
ax.text(-1175, 265, r'$\mathcal{A}$', ha='center', fontsize=12,
        color='#d62728', fontweight='bold')
ax.text(635, 275, r'$\mathcal{D}$', ha='center', fontsize=12,
        color='#1f77b4', fontweight='bold')

# Altitude + velocity info boxes
bbox_props = dict(boxstyle="round,pad=0.2", fc="0.97", ec="0.5", lw=0.5)
ax.text(-1400, 305,
        r'$z_A\!\in\![280,320]$\,m,  $V_A\!\in\![45,50]$\,m/s,  $a_{n,A}^{\max}\!=\!3g$',
        fontsize=7.5, ha='left', bbox=bbox_props)
ax.text(370, 305,
        r'$z_D\!\in\![320,380]$\,m,  $V_D\!\in\![55,60]$\,m/s,  $a_{n,D}^{\max}\!=\!5g$',
        fontsize=7.5, ha='left', bbox=bbox_props)

# Common params
ax.text(-1560, -390,
        r'$p_H\!=\!(1200,0,0)^{\!\top}$m,  $\rho^{\rm kill}\!=\!5$\,m,  '
        r'$\alpha_j\!=\!30^\circ$,  $N_{\rm PN}\!=\!3$,  $\Delta t\!=\!0.01$\,s',
        fontsize=7, ha='left', color='0.4')

# ── Coord axis ──
ax.annotate('', xy=(-1530, -300), xytext=(-1530, -240),
            arrowprops=dict(arrowstyle='->', color='k', lw=0.9))
ax.annotate('', xy=(-1470, -300), xytext=(-1530, -300),
            arrowprops=dict(arrowstyle='->', color='k', lw=0.9))
ax.text(-1460, -313, r'$x$', fontsize=9)
ax.text(-1545, -233, r'$y$', fontsize=9)

ax.set_xlim(-1580, 1350)
ax.set_ylim(-400, 340)
ax.set_aspect('equal')
ax.axis('off')

# ── Legend ──
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
           markersize=6, label=r'$A_i$'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#1f77b4',
           markersize=6, label=r'$D_j$'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#2ca02c',
           markersize=10, label=r'HVT'),
]
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.92,
          edgecolor='0.6', fontsize=8, ncol=3, handletextpad=0.3)

plt.tight_layout(pad=0.15)
out = '/home/uav/11gsytset/0A_LYX_CODE/PAPER/swarm/swarmtex/figures/V71_6688-new/fig_scenario_setup.pdf'
plt.savefig(out, dpi=300, bbox_inches='tight')
print("Done:", out)
