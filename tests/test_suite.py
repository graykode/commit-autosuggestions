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
import unittest

class CitiesTestCase(unittest.TestCase):
    endpoint='http://127.0.0.1:5000'
    headers = {'Content-Type': 'application/json; charset=utf-8'}

    def test_index(self):
        response = requests.get(f"{self.endpoint}/", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            json.loads(response.text),
            {
                "hello": "world",
            }
        )

    def test_tokenizer(self):
        response = requests.post(
            f"{self.endpoint}/tokenizer",
            headers=self.headers,
            data=json.dumps(
                dict(
                    code="hello world!"
                )
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            json.loads(response.text),
            {
                "tokens": [
                    "hello",
                    "Ġworld",
                    "!"
                ]
            }
        )

    def test_added(self):
        response = requests.post(
            f"{self.endpoint}/added",
            headers=self.headers,
            data=json.dumps(
                dict(
                    idx=0,
                    added=['test'],
                    deleted=[],
                )
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            json.loads(response.text),
            {'idx': 0, 'message': ['Test method .']}
        )

    def test_added(self):
        response = requests.post(
            f"{self.endpoint}/diff",
            headers=self.headers,
            data=json.dumps(
                dict(
                    idx=0,
                    added=['tes'],
                    deleted=['test'],
                )
            )
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            json.loads(response.text),
            {'idx': 0, 'message': ['Fix typo']}
        )


def suite():
    suties = unittest.TestSuite()
    suties.addTests(unittest.makeSuite(CitiesTestCase))
    return suties