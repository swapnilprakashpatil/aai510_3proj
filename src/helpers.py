import matplotlib.pyplot as plt
import time as time_module
import psutil
import threading

class CPUMonitor(threading.Thread):
    def __init__(self, interval=1):
        threading.Thread.__init__(self)
        self.interval = interval
        self.running = False
        self.cpu_percentages = []
        self.timestamps = []
        
    def run(self):
        self.running = True
        start_time = time_module.time()
        while self.running:
            cpu_percent = psutil.cpu_percent(interval=self.interval)
            self.cpu_percentages.append(cpu_percent)
            self.timestamps.append(time_module.time() - start_time)
            
    def stop(self):
        self.running = False
        
    def get_data(self):
        return self.timestamps, self.cpu_percentages
    
    def plot(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.cpu_percentages)
        plt.title('CPU Usage During Model Training')
        plt.xlabel('Time (seconds)')
        plt.ylabel('CPU Usage (%)')
        plt.ylim(0, 100)
        plt.grid(True)
        plt.show()