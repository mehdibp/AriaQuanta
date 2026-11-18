import os
import timeit
import functools
import concurrent.futures
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from AriaQuanta._utils import np, Config
from AriaQuanta.aqc import circuit
from AriaQuanta.aqc.circuit import Circuit
from AriaQuanta.backend.job import Job
from AriaQuanta.backend.result import Result, ResultDensity

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


SUPPORTED_HARDWARE = ('Local',)   # 'HPC' / 'Cloud' / 'QPU' are declared in Config but not implemented yet


# -------------------------------------------------------------------------------------------
class Simulator:
    def __init__(self) -> None:
        self.circuit   : Optional[Circuit] = None
        self.iterations: int = 0
        self.density   : bool = False


    # ------------------------------------------------------------
    def simulate(self, circuit: Circuit, iterations: int, num_nodes: int=1,
                    density: bool=False, show_progress: bool=True) -> Union[Result, ResultDensity]:
        self._check_validation(circuit, iterations, num_nodes)
        
        self.circuit    = circuit
        self.iterations = iterations
        self.density    = density

        run_one_shot = functools.partial(self._run_one_shot, circuit, density)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_nodes) as executor:
            results_iter: Iterator[Tuple[np.ndarray, Dict[str, str]]] = executor.map(run_one_shot, range(iterations))
            if show_progress:
                results_iter = _with_progress(results_iter, total=iterations)
            state_all, measurequbit_values_all = zip(*results_iter)

        if density:
            return ResultDensity(list(state_all))
        else:
            return Result(list(state_all), circuit.num_of_qubits, circuit.num_of_ancilla, list(measurequbit_values_all))

    # ------------------------------------------------------------
    @staticmethod
    def _run_one_shot(circuit: Circuit, density: bool, job_id: int) -> Tuple[np.ndarray, Dict[str, str]]:
        this_job = Job(str(job_id).zfill(6))
        return this_job.job_run(circuit, density)

    # ------------------------------------------------------------
    @staticmethod
    def _check_validation(circuit: Circuit, iterations: int, num_nodes: int):
        if not isinstance(circuit, Circuit):
            raise TypeError("'circuit' must be a Circuit instance, got {}.".format(type(circuit).__name__))
        if iterations < 1:
            raise ValueError("'iterations' must be at least 1, got {}.".format(iterations))
        if num_nodes  < 1:
            raise ValueError("'num_nodes' must be at least 1, got {}.".format(num_nodes))
        if Config.hardware not in SUPPORTED_HARDWARE:
            raise NotImplementedError(
                "Simulator currently only supports Config.hardware == 'Local' (got {!r}); "
                "'HPC'/'Cloud'/'QPU' are declared but not implemented yet.".format(Config.hardware)
            )
        


# -------------------------------------------------------------------------------------------
def _with_progress(iterable: Iterable[Any], total: int) -> Iterator[Any]:
    if _HAS_TQDM:
        return tqdm(iterable, total=total, desc='Simulating', unit='shot')
    return _SimpleProgress(iterable, total)

# -------------------------------------------------------------------------------------------
class _SimpleProgress:
    # Minimal, dependency-free progress reporter used when tqdm isn't installed

    def __init__(self, iterable: Iterable[Any], total: int) -> None:
        self._iterable = iterable
        self._total = total

    def __iter__(self) -> Iterator[Any]:
        count = 0
        for item in self._iterable:
            count += 1
            print('\rSimulating: {}/{}'.format(count, self._total), end='', flush=True)
            yield item
        print()


# -------------------------------------------------------------------------------------------
def profile_executor(executor_class: type, function: Any, iterable: Iterable[Any], max_workers: int) -> float:
    with executor_class(max_workers=max_workers) as executor:
        start_time = timeit.default_timer()
        list(executor.map(function, iterable))
        elapsed = timeit.default_timer() - start_time
    return elapsed

# -------------------------------------------------------------------------------------------
def choose_best_executor(function: Any, iterable: Iterable[Any], max_workers: Optional[int]=None) -> Tuple[type, float]:
    if max_workers is None:
        max_workers = os.cpu_count()    # Use number of CPU cores as default

    thread_time  = profile_executor(concurrent.futures.ThreadPoolExecutor , function, iterable, max_workers)
    process_time = profile_executor(concurrent.futures.ProcessPoolExecutor, function, iterable, max_workers)

    if thread_time < process_time:
        return concurrent.futures.ThreadPoolExecutor, thread_time
    else:
        return concurrent.futures.ProcessPoolExecutor, process_time

