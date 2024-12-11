import os
from datetime import datetime

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        with open(self.log_file, "w") as file:
            file.write(f"Log file created: {datetime.now().isoformat()}\n")

    def log(self, message):
        print(message)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        with open(self.log_file, "a") as file:
            file.write(log_message)