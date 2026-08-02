import time
from enum import Enum
from typing import Dict, Optional, Tuple

from AriaQuanta._utils import np
from AriaQuanta.aqc.circuit import Circuit


# -------------------------------------------------------------------------------------------
class JobStatus(Enum):
    PENDING     = 'pending'
    RUNNING     = 'running'
    COMPLETED   = 'completed'
    FAILED      = 'failed'

    def __str__(self) -> str:
        return self.value


# -------------------------------------------------------------------------------------------
class Job():
    def __init__(self, job_id: str) -> None:
        if not job_id:
            raise ValueError("'job_id' must be a non-empty string.")

        self.job_id = job_id
        self.status: JobStatus = JobStatus.PENDING
        self.error : Optional[BaseException] = None
        self.start_time: Optional[float] = None
        self.end_time  : Optional[float] = None


    # ------------------------------------------------------------
    def job_run(self, qc: Circuit, density: bool=False) -> Tuple[np.ndarray, Dict[str, str]]:
        if not isinstance(qc, Circuit):
            raise TypeError("'qc' must be a Circuit instance, got {}.".format(type(qc).__name__))

        self.status = JobStatus.RUNNING
        self.start_time = time.perf_counter()

        try:
            this_qc = qc.copy()          # independent copy -- see the thread-safety note above
            state = this_qc.run_density() if density else this_qc.run()
        except Exception as error:
            self.status = JobStatus.FAILED
            self.error = error
            self.end_time = time.perf_counter()
            raise                        # Simulator/executor.map should still see the failure

        self.status = JobStatus.COMPLETED
        self.end_time = time.perf_counter()
        return state, this_qc.measurequbit_values

    # ------------------------------------------------------------
    @property
    def duration(self) -> Optional[float]:
        # seconds, using a monotonic clock (perf_counter) -- None until the job finishes
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    # ------------------------------------------------------------
    def __repr__(self) -> str:
        duration = self.duration
        duration_str = '{:.4f}s'.format(duration) if duration is not None else 'n/a'
        return "Job(id={!r}, status={}, duration={})".format(self.job_id, self.status, duration_str)



