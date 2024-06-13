import threading
import time

class APIManager:
    def __init__(self, max_requests_per_second):
        self.max_requests_per_second = max_requests_per_second
        self.lock = threading.Lock()
        self.last_request_time = 0

    def wait_for_api_limit_restriction(self, worker_id):
        with self.lock:
            current_time = time.time()
            elapsed_time = current_time - self.last_request_time
            if elapsed_time < 1 / self.max_requests_per_second:
                sleep_time = (1 / self.max_requests_per_second) - elapsed_time
                time.sleep(sleep_time)
            self.last_request_time = time.time()

        #print(f"Worker {worker_id} asked for permission @ {current_time} and received permission @ {self.last_request_time}")
