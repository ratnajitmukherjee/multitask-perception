import glob
import os
import sys
from datetime import datetime
from threading import Timer
from time import sleep
from time import time

# TODO: pynvml not installed — pip install pynvml
try:
    import pynvml
except ImportError:
    pynvml = None
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


class EnergyMeter:
    def __init__(
        self, writer: SummaryWriter = None, period=0.01, dir="./", dataset=None
    ):
        assert period >= 0.005, "Measurement period below than 5ms"
        self.period = period
        pynvml.nvmlInit()
        self.gpu_handles = [
            pynvml.nvmlDeviceGetHandleByIndex(idx)
            for idx in range(pynvml.nvmlDeviceGetCount())
        ]
        self.dir = dir
        self.dataset = dataset
        self.writer = writer
        if self.writer is not None:
            self.writer.add_scalar("xtras/energy_usage", 0, 0)
        self.done = False
        self.steps = 0
        self.energy = 0
        self.next_t = 0

    def __enter__(self):
        self.done = False
        self.steps = 0
        self.energy = 0
        self.next_t = time()
        self.run()
        return self

    def _get_energy_usage(self):
        energy = 0
        for handle in self.gpu_handles:
            power = pynvml.nvmlDeviceGetPowerUsage(handle)
            energy += power / 1000.0 * self.period
        return energy

    def run(self):
        if not self.done:
            self.t = Timer(self.next_t - time(), self.run)
            self.t.start()
            self.next_t += self.period
            self.steps = self.steps + 1
            self.energy = self.energy + self._get_energy_usage()
            if self.steps % 100 == 0:
                if self.writer is not None:
                    self.writer.add_scalar(
                        "xtras/energy_usage", self.energy, self.steps
                    )

    def __exit__(self, type, value, traceback):
        self.done = True
        self.t.cancel()
        path = os.path.join(self.dir, "inference", self.dataset)
        os.makedirs(path, exist_ok=True)
        list_of_files = glob.glob(
            path + "**/*.txt"
        )  # * means all if need specific format then *.csv
        result_string = f"\nTotal energy used (in J): {round(self.energy):.2f}"
        time = datetime.now().strftime("%H:%M:%S")
        if list_of_files:
            result_path = max(list_of_files, key=os.path.getmtime)
        else:
            result_path = os.path.join(path, f"test_result_{time}.txt")
        with open(result_path, "w") as file:
            file.write(result_string)
        print(f"Total energy used (in J): {round(self.energy):.2f} \n")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        PERIOD = float(sys.argv[1])
    else:
        PERIOD = 0.01

    em = EnergyMeter(PERIOD)
    with em:
        # put code you what to measure energy of here
        sleep(2)
