# Copyright 2020-present Tae Hwan Jung
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import git
import json
import argparse
from git import Repo
from tqdm import tqdm
from time import sleep
from queue import Queue
from threading import Thread

class ClonePooler(object):
    def __init__(self, total_repos):
        self.count = 0
        self.total_repos = total_repos
        self._queue = Queue()
        self.num_worker_threads = args.num_worker_threads
        self.repos_dir = args.repos_dir

        for i in range(self.num_worker_threads):
            _thread = Thread(target=self._worker)
            _thread.daemon = True
            _thread.start()

    def _worker(self):
        while True:
            repos = self._queue.get()
            self.do_job(repos)
            self._queue.task_done()

    def set_queue(self, repos):
        self._queue.put(repos)

    def join_queue(self):
        self._queue.join()

    def do_job(self, repo):
        try:
            Repo.clone_from(
                f'https://:@github.com/{repo}.git',
                f'{self.repos_dir}/{repo}'
            )
            sleep(0.1)
            self.count += 1
            print(f"{self.count}/{self.total_repos} {(self.count/self.total_repos) * 100}")
        except git.exc.InvalidGitRepositoryError:
            print(f'{repo} is not found.')
        except git.exc.GitError as e:
            print(e)

def main(args):

    os.makedirs(args.repos_dir, exist_ok=True)
    repos = set()
    with open(args.jsonl_file, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            repos.add(js['repo'])

    pooler = ClonePooler(
        total_repos=len(repos)
    )
    for repo in repos:
        pooler.set_queue(repo)
    pooler.join_queue()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--jsonl_file", type=str, required=True,
                        help="jsonl file path.")
    parser.add_argument("--repos_dir", type=str, required=True,
                        help="directory that all repositories will be downloaded.")
    parser.add_argument("--num_worker_threads", type=int, default=16,
                        help="number of threads in a worker")

    args = parser.parse_args()

    main(args)