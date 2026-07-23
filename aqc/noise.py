from AriaQuanta._utils import np
from AriaQuanta.aqc.gatelibrary import X, Y, Z


# -------------------------------------------------------------------------------------------
class NoiseClass:
    def __init__(self, name, noise_gate, probability, target_qubits):
        
        self.name = name
        self.noise_gate = noise_gate    
        self.probability = probability
        self.target_qubits = target_qubits
        self.qubits = [target_qubits]

    def apply(self, num_of_qubits, multistate):

        p = np.random.rand(1)[0]
        if p < self.probability:
            if self.target_qubits == -1:
                q = np.random.randint(0, high=num_of_qubits, size=1, dtype=int)
            else:
                q = self.target_qubits

            this_gate = self.noise_gate
            multistate = this_gate(q).apply(num_of_qubits, multistate)
        else:
            pass


        return multistate
    
    def apply_noise_density(self, num_of_qubits, density_matrix):
        return 1

# -------------------------------------------------------------------------------------------
class BitFlipNoise(NoiseClass):
    def __init__(self, probability=1.0, target_qubits=-1):
        noise_gate = X
        super().__init__(name='BitFlip', noise_gate=noise_gate, probability=probability, target_qubits=target_qubits)
    
# -------------------------------------------------------------------------------------------
class PhaseFlipNoise(NoiseClass):
    def __init__(self, probability=1.0, target_qubits=-1):
        noise_gate = Z
        super().__init__(name='PhaseFlip', noise_gate=noise_gate, probability=probability, target_qubits=target_qubits)

# -------------------------------------------------------------------------------------------
class DepolarizingNoise(NoiseClass):
    def __init__(self, probability=1.0, target_qubits=-1):
        #choose_gate = np.random.choice(['XGate','YGate','ZGate'])

        noise_gate = Y

        #if choose_gate == 'XGate':
        #    noise_gate = X
        #elif choose_gate == 'YGate':
        #    noise_gate = Y
        #elif choose_gate == 'ZGate':
        #    noise_gate = Z

        super().__init__(name='Depolarizing', noise_gate=noise_gate, probability=probability, target_qubits=target_qubits)