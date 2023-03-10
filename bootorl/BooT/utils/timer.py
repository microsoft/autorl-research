# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import time


class Timer:
    def __init__(self, total_num=-1, formatting=True, return_total_time=True, return_eta=True):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.total_num = total_num
        self.cnt = 0

        self.formatting = formatting
        self.return_total_time = return_total_time
        self.return_eta = return_eta and (self.total_num > 0)
        self.prev_time_est = None

    def _fmt(self, time):
        if not self.formatting:
            return f"{time:.3f}"
        minutes, seconds = time // 60, time % 60
        hours, minutes = minutes // 60, minutes % 60
        string = f"{minutes:0>2.0f}:{seconds:0>6.3f}"
        if hours > 0:
            string = f"{hours:0>2.0f}:{string}"
        return string

    def __call__(self):
        self.cnt += 1
        
        curr_time = time.time()
        diff_time = curr_time - self.last_time
        returns = [self._fmt(diff_time)]
        if self.return_total_time:
            returns.append(self._fmt(curr_time - self.start_time))
        if self.return_eta:
            self.prev_time_est = diff_time * 0.5 + self.prev_time_est * 0.5 if self.prev_time_est is not None else diff_time
            if self.cnt > self.total_num or self.cnt == 0:
                eta = "??"
            else:
                eta = self.prev_time_est * (self.total_num - self.cnt)
                eta = self._fmt(eta)
            returns.append(eta)

        self.last_time = curr_time
        return tuple(returns)
