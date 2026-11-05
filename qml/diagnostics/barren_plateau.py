from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from AriaQuanta._utils import np
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.aqc.ansatz import Ansatz
from AriaQuanta.algorithms.eigen_solver import Hamiltonian, parse_pauli_string
from AriaQuanta.qml._shared import PAULI_MATRIX


# -------------------------------------------------------------------------------------------
# Barren-plateau diagnostics for AriaQuanta.qml.
#
# A barren plateau (McClean et al., 2018) is the regime where, for randomly initialized
# parameters, the gradient of the cost function vanishes *exponentially* with the number of
# qubits: Var[dC/dtheta] ~ exp(-a * n). Training then needs exponentially many shots just to
# see which direction is downhill. This module measures that directly:
#
#   gradient_statistics   -- Var[dC/dtheta_k] over many random initializations, for ONE ansatz
#   barren_plateau_scan   -- repeats that while a knob (qubit count, depth, ...) is varied, and
#                            fits  Var ~ A * exp(-rate * x)  to the result
#   plot_barren_plateau   -- semilog plot of one or several scans (e.g. global vs local cost)
#
# Two deliberate design choices:
#   * Everything here uses *exact* statevector expectation values (exact_expectation), never
#     shot-based find_expectation_value. Shot noise adds a variance floor of ~1/shots to every
#     gradient estimate, which would swamp -- and hide -- exactly the exponentially small
#     variances this tool exists to detect.
#   * Gradients come from the parameter-shift rule applied to that raw expectation value, so
#     they are exact too (see AriaQuanta.qml.gradients.parameter_shift_gradient: exact for a
#     raw expectation value, which is precisely what is differentiated here -- never a loss).
#
# Cost: the simulator is dense (2^n x 2^n per gate), so scans are practical up to roughly 8-10
# qubits -- enough to see an exponential trend clearly, which is the point.
# -------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------
# Exact expectation values
# -------------------------------------------------------------------------------------------
def exact_expectation(circuit: Circuit, hamiltonian: Hamiltonian) -> float:
    """
    Exact <psi|H|psi> for the state prepared by `circuit` (run once, no sampling).

    Each Pauli string is applied to the statevector one qubit at a time as a 2x2 contraction
    along that qubit's tensor axis (O(2^n) per term, no 2^n x 2^n operator is ever built).
    """
    if not isinstance(circuit, Circuit):
        raise TypeError("'circuit' must be a Circuit instance, got {}.".format(type(circuit).__name__))
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("'hamiltonian' must be a Hamiltonian instance, got {}.".format(type(hamiltonian).__name__))

    n = circuit.num_of_qubits
    psi = np.asarray(circuit.run(), dtype=complex).reshape([2] * n)   # tensor axis q == qubit q (qubit 0 most significant)

    total = 0.0
    for pauli_string, coef in zip(hamiltonian.paulis, hamiltonian.coefs):
        applied = psi
        for qubit, letter in parse_pauli_string(pauli_string, n):
            applied = np.moveaxis(np.tensordot(PAULI_MATRIX[letter], applied, axes=([1], [qubit])), 0, qubit)
        total += float(coef) * float(np.real(np.vdot(psi, applied)))
    return total


# -------------------------------------------------------------------------------------------
# Ready-made cost Hamiltonians -- the local-vs-global distinction is the classic barren-plateau
# experiment (Cerezo et al., 2021: a *global* cost hits a barren plateau even for shallow
# circuits; a *local* one stays trainable until the circuit gets deep).
# -------------------------------------------------------------------------------------------
def local_z_hamiltonian(num_of_qubits: int, qubit: int = 0) -> Hamiltonian:
    """H = Z_qubit -- a local (single-qubit) cost."""
    if not (0 <= qubit < num_of_qubits):
        raise ValueError("'qubit' must be in [0, {}), got {}.".format(num_of_qubits, qubit))
    return Hamiltonian([('Z{}'.format(qubit), 1.0)])

def global_z_hamiltonian(num_of_qubits: int) -> Hamiltonian:
    """H = Z_0 Z_1 ... Z_{n-1} -- a global (all-qubit parity) cost."""
    if num_of_qubits < 1:
        raise ValueError("'num_of_qubits' must be at least 1, got {}.".format(num_of_qubits))
    return Hamiltonian([(''.join('Z{}'.format(q) for q in range(num_of_qubits)), 1.0)])


# -------------------------------------------------------------------------------------------
# Gradient statistics for one ansatz
# -------------------------------------------------------------------------------------------
@dataclass
class GradientStatistics:
    """
    Result of gradient_statistics().

    :ivar gradients: shape (num_of_samples, len(param_indices)) -- dC/dtheta_k for every
                     sampled initialization and every inspected parameter k.
    :ivar init_values: shape (num_of_samples, num_of_params) -- the sampled initializations
                     (kept so any single result can be reproduced / cross-checked).
    """
    num_of_qubits: int
    num_of_params: int
    num_of_samples: int
    param_indices: List[int]
    gradients: np.ndarray
    init_values: np.ndarray

    @property
    def variance_per_param(self) -> np.ndarray:
        return np.var(self.gradients, axis=0, ddof=1)

    @property
    def mean_variance(self) -> float:
        """Variance averaged over the inspected parameters -- the scan's headline number."""
        return float(np.mean(self.variance_per_param))

    @property
    def mean_abs_gradient(self) -> float:
        return float(np.mean(np.abs(self.gradients)))

    def __repr__(self) -> str:
        return ("GradientStatistics(num_of_qubits={}, num_of_params={}, num_of_samples={}, "
                "param_indices={}, mean_variance={:.3e})"
                .format(self.num_of_qubits, self.num_of_params, self.num_of_samples,
                        self.param_indices, self.mean_variance))


def gradient_statistics(ansatz: Ansatz, hamiltonian: Hamiltonian, num_of_samples: int = 50,
                        param_indices: Optional[Union[int, Sequence[int]]] = None,
                        init_range: Tuple[float, float] = (0.0, 2.0 * np.pi),
                        seed: Optional[int] = None) -> GradientStatistics:
    """
    Samples `num_of_samples` random parameter vectors (uniform in `init_range` -- the
    uniform-over-the-circle initialization the barren-plateau result is stated for) and, at
    each, computes the exact gradient of <H> with respect to the chosen parameter(s) via the
    parameter-shift rule.

    :param ansatz: Ansatz to probe. Its parameters are restored to their original values
                    before this function returns.
    :param hamiltonian: Cost observable, e.g. local_z_hamiltonian(n) or global_z_hamiltonian(n).
    :param num_of_samples: Number of random initializations (>= 2; more -> a steadier variance
                    estimate, whose relative error is about sqrt(2 / num_of_samples)).
    :param param_indices: Which parameter(s) to differentiate -- an int or a list of indices
                    into ansatz.params_names. Default: first, middle and last parameter
                    (de-duplicated), so the number isn't dominated by one layer's behaviour.
    :param init_range: (low, high) of the uniform initialization distribution.
    :param seed: Seeds the sampling, for reproducibility.
    """
    _check_validation(ansatz, hamiltonian, num_of_samples, init_range)
    n_params = len(ansatz.params_names)
    indices = _resolve_param_indices(param_indices, n_params)

    rng = np.random.default_rng(seed)
    original_values = np.array(ansatz.params_values, dtype=float).copy()

    init_values = rng.uniform(init_range[0], init_range[1], size=(num_of_samples, n_params))
    gradients = np.zeros((num_of_samples, len(indices)))
    try:
        for s in range(num_of_samples):
            for j, k in enumerate(indices):
                gradients[s, j] = _single_parameter_shift(ansatz, hamiltonian, init_values[s], k)
    finally:
        ansatz.set_params_values(original_values)

    return GradientStatistics(num_of_qubits=ansatz.num_of_qubits, num_of_params=n_params,
                              num_of_samples=num_of_samples, param_indices=indices,
                              gradients=gradients, init_values=init_values)


# -------------------------------------------------------------------------------------------
# Scans: vary a knob (qubit count, depth, ...) and watch the variance
# -------------------------------------------------------------------------------------------
class BarrenPlateauScan:
    """
    Gradient statistics collected while one knob (`scan_name`: qubit count by default) is swept.

    :ivar scan_values: the swept values (e.g. [2, 3, 4, ...]).
    :ivar stats: one GradientStatistics per scan value.
    """

    def __init__(self, scan_values: Sequence[float], stats: Sequence[GradientStatistics],
                 label: str = '', scan_name: str = 'num_of_qubits') -> None:
        self.scan_values = np.asarray(scan_values, dtype=float)
        self.stats = list(stats)
        self.label = label
        self.scan_name = scan_name

    @property
    def variances(self) -> np.ndarray:
        """Mean gradient variance at every scan value."""
        return np.array([s.mean_variance for s in self.stats])

    def fit_exponential_decay(self) -> Dict[str, float]:
        """
        Least-squares fit of  ln(Var) = ln(A) - rate * x  over the scan:  Var ~ A * exp(-rate * x).

        Returns {'rate', 'prefactor', 'base', 'r_squared'} where base = exp(rate) -- so the
        textbook barren plateau Var ~ 2^-n shows up as rate ~ 0.69 (base ~ 2). A rate near 0
        (or negative) means no exponential decay over the scanned range. Scan values whose
        variance is exactly 0 are dropped from the fit; at least two are needed.
        """
        x, v = self.scan_values, self.variances
        keep = v > 0
        if np.count_nonzero(keep) < 2:
            raise ValueError("Need at least two scan values with non-zero variance to fit a decay, got {}.".format(int(np.count_nonzero(keep))))
        x, y = x[keep], np.log(v[keep])
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        ss_res = float(np.sum((y - predicted) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rate = float(-slope)
        return {'rate': rate, 'prefactor': float(np.exp(intercept)),
                'base': float(np.exp(rate)), 'r_squared': float(r_squared)}

    def summary(self) -> str:
        """A printable table plus the fitted decay -- the text version of plot_barren_plateau."""
        lines = ["Barren-plateau scan{} (variance of dC/dtheta over {} random initializations)".format(
            ' "{}"'.format(self.label) if self.label else '', self.stats[0].num_of_samples if self.stats else 0)]
        lines.append("{:>16s} {:>10s} {:>14s} {:>14s}".format(self.scan_name, 'n_params', 'mean Var', 'mean |grad|'))
        for x, s in zip(self.scan_values, self.stats):
            lines.append("{:>16g} {:>10d} {:>14.3e} {:>14.3e}".format(x, s.num_of_params, s.mean_variance, s.mean_abs_gradient))
        try:
            fit = self.fit_exponential_decay()
            lines.append("fit: Var ~ {:.3g} * exp(-{:.3f} * {})   [base {:.2f}^-x, R^2 = {:.3f}]".format(
                fit['prefactor'], fit['rate'], self.scan_name, fit['base'], fit['r_squared']))
        except ValueError as error:
            lines.append("fit unavailable: {}".format(error))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "BarrenPlateauScan(label={!r}, {}={}, variances={})".format(
            self.label, self.scan_name, self.scan_values.tolist(), np.array2string(self.variances, precision=3))


def barren_plateau_scan(ansatz_factory: Callable[[int], Ansatz],
                        hamiltonian_factory: Union[Callable[[int], Hamiltonian], Hamiltonian],
                        scan_values: Sequence[int], num_of_samples: int = 50,
                        param_indices: Optional[Union[int, Sequence[int]]] = None,
                        init_range: Tuple[float, float] = (0.0, 2.0 * np.pi),
                        seed: Optional[int] = None, label: str = '',
                        scan_name: str = 'num_of_qubits', verbose: bool = False) -> BarrenPlateauScan:
    """
    Runs gradient_statistics() once per scan value, building a fresh ansatz/Hamiltonian each
    time from the two factories (each takes the scan value).

    Typical use -- scan the qubit count with depth growing alongside it:
        scan = barren_plateau_scan(
            ansatz_factory      = lambda n: HardwareEfficientAnsatz(n, rotation='RY', reps=n),
            hamiltonian_factory = global_z_hamiltonian,
            scan_values         = [2, 3, 4, 5, 6])
        print(scan.summary())

    Or scan depth at a fixed width by making the factory ignore nothing but use the value as `reps`
    (and pass scan_name='reps'). `hamiltonian_factory` may also be a single Hamiltonian reused
    for every scan value (only valid when it fits every ansatz in the scan).
    """
    if not callable(ansatz_factory):
        raise TypeError("'ansatz_factory' must be callable, got {}.".format(type(ansatz_factory).__name__))
    if not (callable(hamiltonian_factory) or isinstance(hamiltonian_factory, Hamiltonian)):
        raise TypeError("'hamiltonian_factory' must be a Hamiltonian or a callable returning one, got {}.".format(type(hamiltonian_factory).__name__))
    scan_values = list(scan_values)
    if len(scan_values) == 0:
        raise ValueError("'scan_values' must contain at least one value.")

    seeds = np.random.SeedSequence(seed).spawn(len(scan_values))   # independent, reproducible stream per scan value
    stats: List[GradientStatistics] = []
    for value, child in zip(scan_values, seeds):
        ansatz = ansatz_factory(value)
        hamiltonian = hamiltonian_factory(value) if callable(hamiltonian_factory) else hamiltonian_factory
        s = gradient_statistics(ansatz, hamiltonian, num_of_samples=num_of_samples, param_indices=param_indices,
                                init_range=init_range, seed=int(child.generate_state(1)[0]))
        stats.append(s)
        if verbose:
            print("{}={}: n_params={}, mean Var = {:.3e}".format(scan_name, value, s.num_of_params, s.mean_variance))

    return BarrenPlateauScan(scan_values, stats, label=label, scan_name=scan_name)


# -------------------------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------------------------
def plot_barren_plateau(scans: Union[BarrenPlateauScan, Sequence[BarrenPlateauScan], Dict[str, BarrenPlateauScan]],
                        ax: Optional[Axes] = None, show_fit: bool = True,
                        reference_base: Optional[float] = 2.0, title: Optional[str] = None,
                        figsize: Tuple[float, float] = (7, 5)) -> Tuple[Figure, Axes]:
    """
    Semilog plot of gradient variance versus the scanned knob, one curve per scan (compare e.g.
    a global and a local cost on the same axes). A straight line on this plot IS exponential
    decay; the slope is the fitted rate.

    :param scans: one BarrenPlateauScan, a list of them, or a {label: scan} dict.
    :param show_fit: also draw each scan's fitted exponential as a dashed line.
    :param reference_base: if given, draws a dotted guide  base^-x  anchored at the first scan's
                    first point (2.0 -> the textbook 2^-n barren-plateau scaling). None to omit.
    """
    if isinstance(scans, BarrenPlateauScan):
        scan_list = [scans]
    elif isinstance(scans, dict):
        scan_list = []
        for name, scan in scans.items():
            scan.label = scan.label or name
            scan_list.append(scan)
    else:
        scan_list = list(scans)
    if not scan_list:
        raise ValueError("'scans' must contain at least one BarrenPlateauScan.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for scan in scan_list:
        line, = ax.semilogy(scan.scan_values, scan.variances, marker='o', label=scan.label or None)
        if show_fit:
            try:
                fit = scan.fit_exponential_decay()
                xs = np.linspace(scan.scan_values.min(), scan.scan_values.max(), 100)
                ax.semilogy(xs, fit['prefactor'] * np.exp(-fit['rate'] * xs), linestyle='--', color=line.get_color(), alpha=0.6,
                            label='fit: rate {:.2f}'.format(fit['rate']))
            except ValueError:
                pass

    if reference_base is not None:
        first = scan_list[0]
        x0, y0 = first.scan_values[0], first.variances[0]
        xs = np.linspace(first.scan_values.min(), first.scan_values.max(), 100)
        ax.semilogy(xs, y0 * float(reference_base) ** (-(xs - x0)), linestyle=':', color='gray',
                    label='{:g}^-x guide'.format(reference_base))

    ax.set_xlabel(scan_list[0].scan_name)
    ax.set_ylabel('Var[dC/dθ]')
    ax.set_title(title or 'Barren-plateau scan: gradient variance')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.25)
    return fig, ax


# validations / helpers ----------------------------------------------------------------------
def _single_parameter_shift(ansatz: Ansatz, hamiltonian: Hamiltonian, base_values: np.ndarray, index: int) -> float:
    # exact d<H>/d(theta_index) at base_values: ( f(theta + pi/2) - f(theta - pi/2) ) / 2.
    # Same rule as AriaQuanta.qml.gradients.parameter_shift_gradient, restricted to one parameter
    # (that function always differentiates all of them -- 2 * n_params evaluations -- which would
    # make scans needlessly expensive when only a few parameters are inspected).
    plus = base_values.copy();  plus[index] += np.pi / 2
    minus = base_values.copy(); minus[index] -= np.pi / 2
    ansatz.set_params_values(plus)
    f_plus = exact_expectation(ansatz, hamiltonian)
    ansatz.set_params_values(minus)
    f_minus = exact_expectation(ansatz, hamiltonian)
    return (f_plus - f_minus) / 2.0

def _resolve_param_indices(param_indices: Optional[Union[int, Sequence[int]]], n_params: int) -> List[int]:
    if param_indices is None:
        return sorted({0, n_params // 2, n_params - 1})
    indices = [int(param_indices)] if np.ndim(param_indices) == 0 else [int(i) for i in param_indices]
    if len(indices) == 0:
        raise ValueError("'param_indices' must contain at least one index.")
    for i in indices:
        if not (0 <= i < n_params):
            raise ValueError("param index {} is out of range for an ansatz with {} parameters.".format(i, n_params))
    return indices

def _check_validation(ansatz, hamiltonian, num_of_samples, init_range) -> None:
    if not isinstance(ansatz, Ansatz):
        raise TypeError("'ansatz' must be an Ansatz instance, got {}.".format(type(ansatz).__name__))
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("'hamiltonian' must be a Hamiltonian instance, got {}.".format(type(hamiltonian).__name__))
    if not isinstance(num_of_samples, int) or isinstance(num_of_samples, bool) or num_of_samples < 2:
        raise ValueError("'num_of_samples' must be an int >= 2 (a variance needs at least two samples), got {}.".format(num_of_samples))
    if len(init_range) != 2 or not init_range[0] < init_range[1]:
        raise ValueError("'init_range' must be (low, high) with low < high, got {}.".format(init_range))
