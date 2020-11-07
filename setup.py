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
from setuptools import setup, find_packages

project_name = "commit"
version = os.environ.get('COMMIT_VERSION', '0.0.0')

if __name__ == "__main__":

    with open('README.md', 'r') as t:
        README = t.read()

    setup(
        # Project Name, Version
        name=project_name,
        version=version,
        long_description=README,
        long_description_content_type='text/markdown',
        # Author
        license="Apache License, Version 2.0",
        author="TaeHwan-Jung",
        author_email="nlkey2022@gmail.com",
        description="A tool that AI automatically recommends commit messages.",
        url="https://github.com/graykode/commit-autosuggestions",
        # Platform, Requires
        python_requires=">=3.5",
        platforms=["any"],
        project_urls={
            "Source Code": "https://github.com/graykode/commit-autosuggestions",
        },
        install_requires = [
            'click>=7.1.2',
            'gitpython>=3.1.7',
            'whatthepatch>=1.0.0',
            'packaging>=20.4',
        ],
        entry_points={
            'console_scripts': [
                'commit=commit.commit:cli'
            ],
        },
        packages=find_packages(),
    )