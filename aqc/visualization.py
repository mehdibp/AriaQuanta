from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from AriaQuanta._utils import np
from AriaQuanta.aqc.measure import Measure
from AriaQuanta.aqc.operations import Operations
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.gatelibrary.gatebase import GateBase


# -------------------------------------------------------------------------------------------
DEFAULT_STYLE: Dict[str, Any] = {
    'color_single':     'lightblue',    # single-qubit gate boxes (H, X, RX, ...)
    'color_2q_box':     '#DDA0DD',      # boxed 2-qubit gates (RXX/RYY/RZZ/RXY, Barenco, Givens, ...)
    'color_swap':       'm',            # SWAP-family markers/lines
    'color_toffoli':    'k',            # CCX / RCCX / CSWAP
    'color_control':    '#407a28',      # CX/CZ/CP/CS/CSX/CRX/CRY/CRZ/CNX/CNY/CNZ/CNP
    'color_cu':         '#ffa621',      # CU / CNU
    'color_custom':     '#ebe2c7',      # Custom gate box
    'color_measure':    '#ededed',      # Measure box background
    'color_wire':       'k',            # quantum wire
    'color_clbit_wire': 'gray',         # classical (double-line) wire
    'color_if_true':    'black',        # If_cbit connector, expected value == 1 -> filled dot
    'color_if_false':   'white',        # If_cbit connector, expected value == 0 -> open dot
    'fontsize':         12,
    'small_fontsize':   10,
    'figsize_scale_x':  1.4,
    'figsize_scale_y':  0.9,
}


# -------------------------------------------------------------------------------------------
class CircuitVisualizer:
    gate_plotters: Dict[str, Callable] = {}

    def __init__(self, circuit: Circuit, style: Optional[Dict[str, Any]] = None) -> None:
        self.circuit = circuit
        self.style: Dict[str, Any] = self._resolve_style(style)
        self.fig: Optional[Figure] = None
        self.ax:  Optional[Axes]   = None


    # ------------------------------------------------------------
    def visualize(self, save_path: Optional[str] = None, show: bool = True) -> Tuple[Figure, Axes]:
        circuit = self.circuit
        gates   = circuit.gates
        num_of_qubits = circuit.num_of_qubits
        style = self.style

        clbit_row = self._classical_bit_rows(gates)
        num_of_clbit_rows = len(clbit_row)
        total_rows = num_of_qubits + num_of_clbit_rows

        if not gates:
            fig, ax = plt.subplots(figsize=(2 * style['figsize_scale_x'], total_rows * style['figsize_scale_y']))
            self._draw_wires(ax, [-0.5, 0.5], num_of_qubits, clbit_row, style)
            self._finalize(fig, ax, save_path, show)
            return fig, ax

        x_ids = self._assign_columns(gates)
        max_col = max(x_ids)

        fig, ax = plt.subplots(figsize=((max_col + 2) * style['figsize_scale_x'], total_rows * style['figsize_scale_y']))

        xx = [min(x_ids) - 0.5] + x_ids + [max(x_ids) + 0.5]
        self._draw_wires(ax, xx, num_of_qubits, clbit_row, style)

        for i, gate_i in enumerate(gates):
            gate_i_name = gate_i.name
            condition_info = None

            if isinstance(gate_i, Operations):
                condition_info = (gate_i_name, gate_i.conditions)
                gate_i = gate_i.operation_gate
                gate_i_name = gate_i.name        # re-resolve the name AFTER unwrapping

            plot_func = self.gate_plotters.get(gate_i_name, plot_default)
            plot_func(ax, x_ids[i], gate_i, style)

            if isinstance(gate_i, Measure):
                self._draw_measure_connector(ax, x_ids[i], gate_i, clbit_row, style)
            if condition_info is not None:
                self._draw_condition_connector(ax, x_ids[i], gate_i, condition_info, clbit_row, style)

        self._finalize(fig, ax, save_path, show)
        return fig, ax

    # ------------------------------------------------------------
    def save(self, path: str, dpi: int = 300, **kwargs: Any) -> None:
        if self.fig is None:
            raise RuntimeError("Nothing to save yet -- call visualize() first.")
        self.fig.savefig(path, dpi=dpi, bbox_inches='tight', **kwargs)


    # ------------------------------------------------------------
    @staticmethod
    def _resolve_style(style: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        resolved = dict(DEFAULT_STYLE)
        if style: resolved.update(style)
        return resolved

    @staticmethod
    def _effective_clbits(gate: GateBase) -> List[str]:
        clbits = getattr(gate, 'clbits', None)
        if clbits is None:
            return ['c' + str(q) for q in gate.qubits]
        return [str(c) for c in clbits]

    # ------------------------------------------------------------
    def _assign_columns(self, gates: List[GateBase]) -> List[int]:
        num_of_qubits = self.circuit.num_of_qubits
        next_free_column = [0] * num_of_qubits

        x_ids: List[int] = []
        for gate in gates:
            qubits = gate.qubits
            span = range(min(qubits), max(qubits) + 1)
            column = max(next_free_column[q] for q in span)
            x_ids.append(column)
            for q in span:
                next_free_column[q] = column + 1

        return x_ids

    # ------------------------------------------------------------
    def _classical_bit_rows(self, gates: List[GateBase]) -> Dict[str, int]:
        num_of_qubits = self.circuit.num_of_qubits
        clbit_names: List[str] = []
        for gate in gates:
            if isinstance(gate, Measure):
                for name in self._effective_clbits(gate):
                    if name not in clbit_names:
                        clbit_names.append(name)

        num_rows = max(self.circuit.num_of_clbits, len(clbit_names))
        
        for i in range(num_rows - len(clbit_names)):
            clbit_names.append('c{}'.format(len(clbit_names) + i))

        return {name: num_of_qubits + row for row, name in enumerate(clbit_names)}

    # ------------------------------------------------------------
    def _draw_wires(self, ax: Axes, xx: List[float], num_of_qubits: int, clbit_row: Dict[str, int], style: Dict[str, Any]) -> None:
        xx_arr = np.array(xx)

        for i in range(num_of_qubits):
            row_i = np.ones(xx_arr.shape) * i
            ax.plot(xx_arr, row_i, '-', color=style['color_wire'])
            ax.text(xx_arr[0] - 0.5, i, 'Q{}:'.format(i), fontsize=style['fontsize'], ha='center', va='center')

        for name, row in clbit_row.items():
            row_arr = np.ones(xx_arr.shape) * row
            # a pair of close parallel lines is the standard symbol for a classical wire
            ax.plot(xx_arr, row_arr - 0.03, '-', color=style['color_clbit_wire'], linewidth=1)
            ax.plot(xx_arr, row_arr + 0.03, '-', color=style['color_clbit_wire'], linewidth=1)
            ax.text(xx_arr[0] - 0.5, row, '{}:'.format(name), fontsize=style['small_fontsize'],
                    ha='center', va='center', color=style['color_clbit_wire'])

        ax.invert_yaxis()
        plt.xlabel('Gate Sequence')
        plt.ylabel('Qubits')
        plt.axis('off')
        ax.margins(x=0.15, y=0.15)

    # ------------------------------------------------------------
    def _draw_measure_connector(self, ax: Axes, i: float, gate_i: GateBase, clbit_row: Dict[str, int], style: Dict[str, Any]) -> None:
        for q, c in zip(gate_i.qubits, self._effective_clbits(gate_i)):
            row = clbit_row.get(c)
            if row is None:
                continue
            ax.plot([i, i], [q + 0.4, row - 0.08], '-', color=style['color_clbit_wire'], linewidth=1)
            ax.plot(i, row - 0.08, marker='v', color=style['color_clbit_wire'], markersize=6)

    # ------------------------------------------------------------
    def _draw_condition_connector(self, ax: Axes, i: float, gate_i: GateBase, condition_info: Tuple[str, Any], clbit_row: Dict[str, int], style: Dict[str, Any]) -> None:
        wrapper_name, conditions = condition_info

        if wrapper_name == 'ClassicalControl' or conditions is None:
            # an arbitrary predicate isn't tied to one classical bit -- just flag it
            ax.text(i, min(gate_i.qubits) - 0.55, 'CC', fontsize=style['small_fontsize'], ha='center', va='center', color=style['color_clbit_wire'])
            return

        clbit_name, expected_value = conditions
        row = clbit_row.get(clbit_name)
        if row is None:
            return

        bottom_qubit = max(gate_i.qubits)
        ax.plot([i, i], [bottom_qubit + 0.4, row], '--', color=style['color_clbit_wire'], linewidth=1)

        want_one = str(expected_value) == '1'
        is_true_branch = (wrapper_name == 'If_cbit')
        filled = want_one if is_true_branch else (not want_one)
        facecolor = style['color_if_true'] if filled else style['color_if_false']
        ax.plot(i, row, 'o', markersize=8, markerfacecolor=facecolor, markeredgecolor=style['color_clbit_wire'])

    # ------------------------------------------------------------
    def _finalize(self, fig: Figure, ax: Axes, save_path: Optional[str], show: bool) -> None:
        self.fig, self.ax = fig, ax
        if save_path is not None: self.save(save_path)
        if show: plt.show()




# -------------------------------------------------------------------------------------------
def register_gate_plotter(gate_name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        CircuitVisualizer.gate_plotters[gate_name] = func
        return func
    return decorator


# General Plots -----------------------------------------------------------------------------
def plot_default(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    for q in gate_i.qubits:
        ax.text(i, q, gate_i.name, fontsize=style['fontsize'], ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor=style['color_single']))




# Gate Single Qubit -------------------------------------------------------------------------
# Default: I, X, Y, Z, H, S, T --------------------------------------------------------------
# Others : Ph, Xsqrt, P, RX, RY, RZ, Rot ----------------------------------------------------

# -------------------------------------------------------------------------------------------
@register_gate_plotter('G-Ph')
def plot_gph(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    phase = gate_i.phase
    this_text = '{}({:0.2f})'.format(gate_i.name, phase)
    for q in gate_i.qubits:
        ax.text(i, q, this_text, fontsize=style['fontsize'], ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor=style['color_single']))

# -------------------------------------------------------------------------------------------
@register_gate_plotter('Xsqrt')
def plot_xsqrt(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    this_text = r'$\sqrt{X}$'
    for q in gate_i.qubits:
        ax.text(i, q, this_text, fontsize=style['fontsize'], ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor=style['color_single']))

# -------------------------------------------------------------------------------------------
@register_gate_plotter('P')
def plot_p(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    this_text = '{}({:0.2f})'.format(gate_i.name, gate_i.phase)
    for q in gate_i.qubits:
        ax.text(i, q, this_text, fontsize=style['fontsize'], ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor=style['color_single']))

# -------------------------------------------------------------------------------------------
@register_gate_plotter('RX')
@register_gate_plotter('RY')
@register_gate_plotter('RZ')
def plot_axis_rotation(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    phase = gate_i.phase
    gate_i_type = gate_i.name[1:]
    name = '$R_{%s}$' % (gate_i_type)

    this_text = '{}\n({})'.format(name, phase) if isinstance(phase, str) else '{}\n({:0.2f})'.format(name, phase)

    for q in gate_i.qubits:
        ax.text(i, q, this_text, fontsize=style['fontsize'], ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor=style['color_single']))

# -------------------------------------------------------------------------------------------
@register_gate_plotter('Rot')
def plot_rot(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    theta   = gate_i.theta
    phi     = gate_i.phi
    lambda_ = gate_i.lambda_
    this_text = '{}\n({:0.2f},{:0.2f},{:0.2f})'.format('U', theta, phi, lambda_)
    for q in gate_i.qubits:
        ax.text(i, q, this_text, fontsize=style['fontsize'] - 1, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor=style['color_single']))


# Gate Double Qubit -------------------------------------------------------------------------
# SWAP, ISWAP, SWAPsqrt, ISWAPsqrt, SWAPalpha -----------------------------------------------

# -------------------------------------------------------------------------------------------
@register_gate_plotter('SWAP')
def plot_swap(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    c = style['color_swap']
    ax.plot([i, i], q, c)
    ax.plot(i, q[0], 'x', color=c, markersize=14)
    ax.plot(i, q[1], 'x', color=c, markerfacecolor='None', markersize=14)

# -------------------------------------------------------------------------------------------
@register_gate_plotter('ISWAP')
def plot_iswap(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    c = style['color_swap']
    ax.plot([i, i], q, c)
    for qq in q:
        ax.plot(i, qq, 's', color=c, markersize=14)
        ax.plot(i, qq, 'wx', markerfacecolor='None', markersize=14)

# -------------------------------------------------------------------------------------------
@register_gate_plotter('SWAPsqrt')
def plot_swapsqrt(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    c = style['color_swap']
    ax.plot([i, i], q, c)
    ax.plot(i, q[0], 'x', color=c, markersize=14)
    ax.plot(i, q[1], 'x', color=c, markerfacecolor='None', markersize=14)
    ax.text(i, (q[0] + q[1]) / 2, '1/2', fontsize=style['small_fontsize'], ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.5', edgecolor='black', facecolor=style['color_2q_box']))

# -------------------------------------------------------------------------------------------
@register_gate_plotter('ISWAPsqrt')
def plot_iswapsqrt(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    c = style['color_swap']
    ax.plot([i, i], q, c)
    for qq in q:
        ax.plot(i, qq, 's', color=c, markersize=14)
        ax.plot(i, qq, 'wx', markerfacecolor='None', markersize=14)
    ax.text(i, (q[0] + q[1]) / 2, '1/2', fontsize=style['small_fontsize'], ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.5', edgecolor='black', facecolor=style['color_2q_box']))

# -------------------------------------------------------------------------------------------
@register_gate_plotter('SWAPalpha')
def plot_swapalpha(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    c = style['color_swap']
    alpha = gate_i.alpha
    ax.plot([i, i], q, c)
    ax.plot(i, q[0], 'x', color=c, markersize=14)
    ax.plot(i, q[1], 'x', color=c, markerfacecolor='None', markersize=14)
    ax.text(i, (q[0] + q[1]) / 2, '{:0.2f}'.format(alpha), fontsize=style['small_fontsize'],
            ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.5', edgecolor='black', facecolor=style['color_2q_box']))


# Rotational Gates --------------------------------------------------------------------------
# RXX, RYY, RZZ, RXY ------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
@register_gate_plotter('RXX')
@register_gate_plotter('RYY')
@register_gate_plotter('RZZ')
@register_gate_plotter('RXY')
def plot_2q_rotation(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    phase = gate_i.phase
    name = '$R_{%s}$' % (gate_i.name[1:])

    if isinstance(phase, str):
        this_text_1 = '%s' % (name)
        this_text_2 = r'(%s)' % (phase)
    else:
        angle_phi = round(phase / np.pi, 2)
        this_text_1 = '%s' % (name)
        this_text_2 = r'(%.2f$\pi$)' % (angle_phi)

    q_min, q_max = min(q), max(q)
    height = abs(q[1] - q[0]) + 0.8
    width = 0.8
    rect = Rectangle((i - 0.4, q_min - 0.4), width, height, facecolor=style['color_2q_box'], zorder=2, edgecolor='k')
    ax.add_patch(rect)

    ax.text(i - 0.4 + 0.5 * width, q_min - 0.1, str(q_min), ha='center', va='top', fontsize=style['fontsize'], color='k')
    ax.text(i - 0.4 + 0.5 * width, (q_min + q_max) / 2, this_text_1, ha='center', va='center', fontsize=style['fontsize'], color='k')
    ax.text(i - 0.4 + 0.5 * width, (q_min + q_max) / 2 + 0.3, this_text_2, ha='center', va='center', fontsize=style['fontsize'], color='k')
    ax.text(i - 0.4 + 0.5 * width, q_max + 0.1, str(q_max), ha='center', va='bottom', fontsize=style['fontsize'], color='k')


# Other 2-qubit Gates -----------------------------------------------------------------------
# Barenco, Berkeley, Canonical, Givens, Magic -----------------------------------------------

# -------------------------------------------------------------------------------------------
@register_gate_plotter('Barenco')
@register_gate_plotter('Berkeley')
@register_gate_plotter('Canonical')
@register_gate_plotter('Givens')
@register_gate_plotter('Magic')
def plot_2q_box(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    abbr = {'Barenco': 'Brn', 'Berkeley': 'Brk', 'Canonical': 'Can', 'Givens': 'Gvn', 'Magic': 'Mgc'}
    this_text = abbr[gate_i.name]

    q_min, q_max = min(q), max(q)
    height = abs(q[1] - q[0]) + 0.8
    width = 0.8
    rect = Rectangle((i - 0.4, q_min - 0.4), width, height, facecolor=style['color_2q_box'], zorder=2, edgecolor='k')
    ax.add_patch(rect)

    ax.text(i - 0.4 + 0.5 * width, q_min - 0.1, str(q_min), ha='center', va='top', fontsize=style['fontsize'], color='k')
    ax.text(i - 0.4 + 0.5 * width, (q_min + q_max) / 2, this_text, ha='center', va='center', fontsize=style['fontsize'], color='k')
    ax.text(i - 0.4 + 0.5 * width, q_max + 0.1, str(q_max), ha='center', va='bottom', fontsize=style['fontsize'], color='k')


# Gate Triple Qubit--------------------------------------------------------------------------
# CCX (Toffoli), RCCX (Margolus), CSWAP(Fredkin) --------------------------------------------

# -------------------------------------------------------------------------------------------
@register_gate_plotter('CCX')
def plot_ccx(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    c = style['color_toffoli']
    ax.plot(i, q[0], 'o', color=c, markersize=14)
    ax.plot(i, q[1], 'o', color=c, markersize=14)
    ax.plot(i, q[2], 'o', color=c, markerfacecolor='None', markersize=20)
    ax.plot(i, q[2], '+', color=c, markerfacecolor='None', markersize=20)
    ax.plot([i, i, i], q, color=c)

# -------------------------------------------------------------------------------------------
@register_gate_plotter('RCCX')
def plot_rccx(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    q_sort = sorted(q)

    q_min, q_max = q_sort[0], q_sort[2]
    height = abs(q_max - q_min) + 0.8
    width = 0.8
    rect = Rectangle((i - 0.4, q_min - 0.4), width, height, facecolor='#d4d2d2', zorder=2, edgecolor='k')
    ax.add_patch(rect)

    ax.text(i - 0.4+ 0.1*width, q_sort[0] - 0.1, str(q_sort[0]), ha='left'  , va='top'   , fontsize=style['fontsize'], color='k')
    ax.text(i - 0.4+ 0.1*width, q_sort[1]      , str(q_sort[1]), ha='left'  , va='center', fontsize=style['fontsize'], color='k')
    ax.text(i - 0.4+ 0.1*width, q_sort[2] + 0.1, str(q_sort[2]), ha='left'  , va='bottom', fontsize=style['fontsize'], color='k')
    ax.text(i - 0.4+ 0.6*width, (q_min+q_max)/2, gate_i.name   , ha='center', va='center', fontsize=style['fontsize'], color='k', rotation=90)

# -------------------------------------------------------------------------------------------
@register_gate_plotter('CSWAP')
def plot_cswap(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    c = style['color_toffoli']
    ax.plot(i, q[0], 'o', color=c, markersize=14)
    ax.plot(i, q[1], 'x', color=c, markersize=14)
    ax.plot(i, q[2], 'x', color=c, markersize=14)
    ax.plot([i, i, i], q, color=c)


# Gate Control Qubit ------------------------------------------------------------------------
# CX, CZ, CP, CS, CSX, CU -------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
@register_gate_plotter('CX')  # o---(+)
def plot_cx(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    c = style['color_control']
    ax.plot(i, q[0], 'o', color=c, markersize=12)
    ax.plot(i, q[1], 'o', color=c, markerfacecolor='None', markersize=20)
    ax.plot(i, q[1], '+', color=c, markerfacecolor='None', markersize=20)
    ax.plot([i, i], q, color=c)

# -------------------------------------------------------------------------------------------
@register_gate_plotter('CZ')    # o----o
def plot_cz(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    c = style['color_control']
    ax.plot(i, q[0], 'o', color=c, markersize=12)
    ax.plot(i, q[1], 'o', color=c, markersize=12)
    ax.plot([i, i], q, color=c)

# -------------------------------------------------------------------------------------------
@register_gate_plotter('CP')    # o----P(phi)
def plot_cp(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    phase = gate_i.phase
    c = style['color_control']
    ax.plot(i, q[0], 'o', color=c, markersize=12)
    ax.plot([i, i], q, color=c)

    this_text = '{}\n({})'.format(gate_i.name, phase) if isinstance(phase, str) else '{}\n({:0.2f})'.format(gate_i.name, phase)

    ax.text(i, q[1], this_text, fontsize=style['fontsize'], ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', edgecolor=c, facecolor='white'))

# -------------------------------------------------------------------------------------------
@register_gate_plotter('CS')    # o----S
@register_gate_plotter('CSX')   # o----sqrt(X)
@register_gate_plotter('CRX')
@register_gate_plotter('CRY')
@register_gate_plotter('CRZ')
def plot_control_labeled(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    c = style['color_control']
    name = gate_i.name
    ax.plot(i, q[0], 'o', color=c, markersize=12)
    ax.plot([i, i], q, color=c)

    if name in ('CRX', 'CRY', 'CRZ'):
        axis = name[1:]           # 'RX' / 'RY' / 'RZ'
        phase = gate_i.phase
        this_text = '{}\n({})'.format(axis, phase) if isinstance(phase, str) else '{}\n({:0.2f})'.format(axis, phase)
    elif name == 'CS':
        this_text = 'S'
    else:                          # CSX
        this_text = r'$\sqrt{X}$'

    ax.text(i, q[1], this_text, fontsize=style['fontsize'], ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', edgecolor=c, facecolor='white'))


# -------------------------------------------------------------------------------------------
@register_gate_plotter('CU')
def plot_cu(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    controls = gate_i.control_qubits
    this_text = getattr(gate_i, 'namedraw', gate_i.name)
    c = style['color_cu']

    ax.plot(i, controls, 'o', color=c, markersize=12)
    ax.plot([i, i], q, color=c)

    q_min, q_max = q[1], max(q)
    height = abs(q_max - q_min) + 0.8
    width = 0.8
    rect = Rectangle((i - 0.4, q_min - 0.4), width, height, facecolor=c, zorder=2, edgecolor='k')
    ax.add_patch(rect)

    ax.text(i - 0.4 + 0.5 * width, (q_min + q_max) / 2, this_text, ha='center', va='center', fontsize=style['fontsize'])


# Gate Control-N Qubit (multi-control) ------------------------------------------------------
# CNX, CNY, CNZ, CNP, CNU -------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
@register_gate_plotter('CNX')   # o--o--(+)
def plot_cnx(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    controls = gate_i.control_qubits
    target = gate_i.target_qubits
    c = style['color_control']

    for qc in controls:
        ax.plot(i, qc, 'o', color=c, markersize=12)
    ax.plot(i, target[0], 'o', color=c, markerfacecolor='None', markersize=20)
    ax.plot(i, target[0], '+', color=c, markerfacecolor='None', markersize=20)
    ax.plot([i, i], [min(gate_i.qubits), max(gate_i.qubits)], color=c)

# -------------------------------------------------------------------------------------------
@register_gate_plotter('CNY')
@register_gate_plotter('CNZ')
def plot_cn_labeled(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    controls = gate_i.control_qubits
    target = gate_i.target_qubits
    c = style['color_control']
    label = gate_i.name[2]      # 'Y' or 'Z'

    for qc in controls:
        ax.plot(i, qc, 'o', color=c, markersize=12)
    ax.plot([i, i], [min(gate_i.qubits), max(gate_i.qubits)], color=c)
    ax.text(i, target[0], label, fontsize=style['fontsize'], ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', edgecolor=c, facecolor='white'))

# -------------------------------------------------------------------------------------------
@register_gate_plotter('CNP')
def plot_cnp(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    controls = gate_i.control_qubits
    target = gate_i.target_qubits
    c = style['color_control']
    phase = gate_i.phase

    for qc in controls:
        ax.plot(i, qc, 'o', color=c, markersize=12)
    ax.plot([i, i], [min(gate_i.qubits), max(gate_i.qubits)], color=c)

    this_text = 'P\n({})'.format(phase) if isinstance(phase, str) else 'P\n({:0.2f})'.format(phase)
    ax.text(i, target[0], this_text, fontsize=style['fontsize'], ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.4', edgecolor=c, facecolor='white'))

# -------------------------------------------------------------------------------------------
@register_gate_plotter('CNU')
def plot_cnu(ax: Axes, i: float, gate_i: GateBase, style: Dict[str, Any]) -> None:
    controls = gate_i.control_qubits
    target = gate_i.target_qubits
    this_text = getattr(gate_i, 'namedraw', gate_i.name)
    c = style['color_cu']

    for qc in controls:
        ax.plot(i, qc, 'o', color=c, markersize=12)
    ax.plot([i, i], [min(gate_i.qubits), max(gate_i.qubits)], color=c)

    q_min, q_max = min(target), max(target)
    height = abs(q_max - q_min) + 0.8
    width = 0.8
    rect = Rectangle((i - 0.4, q_min - 0.4), width, height, facecolor=c, zorder=2, edgecolor='k')
    ax.add_patch(rect)
    ax.text(i - 0.4 + 0.5 * width, (q_min + q_max) / 2, this_text, ha='center', va='center', fontsize=style['fontsize'])

    
# Gate Custom -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
@register_gate_plotter('Custom')
def plot_custom(ax: Axes, i: float, gate_i: Any, style: Dict[str, Any]) -> None:
    q = gate_i.qubits
    q_min, q_max = min(q), max(q)
    height = abs(q_max - q_min) + 0.8
    width = 0.8
    rect = Rectangle((i - 0.4, q_min - 0.4), width, height, facecolor=style['color_custom'], zorder=2, edgecolor='k')
    ax.add_patch(rect)
    ax.text(i - 0.4 + 0.5 * width, (q_min + q_max) / 2, 'U', ha='center', va='center', fontsize=style['fontsize'])


# Measurement -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
@register_gate_plotter('MeasureQubitResize')
@register_gate_plotter('MeasureQubit')
def plot_measure(ax: Axes, i: float, gate_i: Any, style: Dict[str, Any]) -> None:
    t = np.linspace(0, 2 * np.pi, 100)
    for q in gate_i.qubits:
        # draw back-to-front: background box, then the meter needle, then the arrow glyph on top
        height = width = 0.8
        rect = Rectangle((i - 0.4, q - 0.4), width, height, facecolor=style['color_measure'], zorder=1, edgecolor='k')
        ax.add_patch(rect)

        u, v = i, q + 0.25    # center of the little dial arc
        a, b = 0.32, 0.22     # x/y radii
        xx = u + a * np.cos(t)
        yy = v + b * np.sin(t)
        idx = yy < v
        ax.plot(xx[idx], yy[idx], color='black', zorder=2)

        ax.text(i, q, u"\u2197", fontsize=27, ha='center', va='center', zorder=3)

