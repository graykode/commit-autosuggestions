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

import json
import requests
import argparse
import subprocess
import whatthepatch

def tokenizing(code):
    data = {"code": code }
    res = requests.post(
        'http://127.0.0.1:5000/tokenizer',
        data=json.dumps(data),
        headers=args.headers
    )
    return json.loads(res.text)["tokens"]

def preprocessing(diffs):
    for idx, example in enumerate(whatthepatch.parse_patch(diffs)):
        isadded, isdeleted = False, False
        added, deleted = [], []
        for change in example.changes:
            if change.old == None and change.new != None:
                added.extend(tokenizing(change.line))
                isadded = True
            elif change.old != None and change.new == None:
                deleted.extend(tokenizing(change.line))
                isdeleted = True

        if isadded and isdeleted:
            data = {"idx": idx, "added" : added, "deleted" : deleted}
            res = requests.post(
                'http://127.0.0.1:5000/diff',
                data=json.dumps(data),
                headers=args.headers
            )
            print(json.loads(res.text))
        else:
            data = {"idx": idx, "added": added, "deleted": deleted}
            res = requests.post(
                'http://127.0.0.1:5000/added',
                data=json.dumps(data),
                headers=args.headers
            )
            print(json.loads(res.text))

def main():

    proc = subprocess.Popen(["git", "diff", "--cached"], stdout=subprocess.PIPE)
    staged_files = proc.stdout.readlines()
    staged_files = [f.decode("utf-8") for f in staged_files]
    staged_files = [f.strip() for f in staged_files]
    diffs = "\n".join(staged_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:5000/")
    args = parser.parse_args()

    args.headers = {'Content-Type': 'application/json; charset=utf-8'}

    main()