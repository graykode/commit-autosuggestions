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

import subprocess
import whatthepatch

def preprocessing(diff):
    added_examples, diff_examples = [], []
    isadded, isdeleted = False, False
    for idx, example in enumerate(whatthepatch.parse_patch(diff)):
        added, deleted = [], []
        for change in example.changes:
            if change.old == None and change.new != None:
                added.extend(tokenizer.tokenize(change.line))
                isadded = True
            elif change.old != None and change.new == None:
                deleted.extend(tokenizer.tokenize(change.line))
                isdeleted = True

    if isadded and isdeleted:
        pass
    else:
        pass

def main():
    proc = subprocess.Popen(["git", "diff", "--cached"], stdout=subprocess.PIPE)
    staged_files = proc.stdout.readlines()
    staged_files = [f.decode("utf-8") for f in staged_files]
    staged_files = [f.strip() for f in staged_files]
    diffs = "\n".join(staged_files)


if __name__ == '__main__':
    main()