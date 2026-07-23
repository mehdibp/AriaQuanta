from AriaQuanta._utils import np


# -------------------------------------------------------------------------------------------
class Job():
    def __init__(self, job_id):
        self.job_id = job_id
        self.status = 'Job %s not yet started'.format(self.job_id)
        
        #self.q_values = {}   # {'0': 1, '1': 1}
        #self.c_values = {}   # {'c0': 1, 'c1': 1}
        #self.qc_values = []  # list of tuples

    def job_run(self, qc, density):

        self.status = 'Job {} started'.format(self.job_id)

        this_qc = qc.copy()
        this_qc.run()

        self.status = 'Job {} completed'.format(self.job_id)

                   
        return this_qc.statevector, this_qc.measurequbit_values
