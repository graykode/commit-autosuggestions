# commit-autosuggestions
<p align="center">
<a href="https://travis-ci.com/github/graykode/commit-autosuggestions"><img alt="Build Status" src="https://travis-ci.com/graykode/commit-autosuggestions.svg?branch=master"></a>
<a href="https://github.com/graykode/commit-autosuggestions/blob/master/LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
<a href="https://pypi.org/project/commit/"><img alt="PyPI" src="https://img.shields.io/pypi/v/commit"></a>
<a href="https://pepy.tech/project/commit"><img alt="Downloads" src="https://static.pepy.tech/badge/commit"></a>
</p>

Have you ever hesitated to write a commit message? Now get a commit message from Artificial Intelligence!

<div align="center">
<span align="left"><img src="https://raw.githubusercontent.com/graykode/commit-autosuggestions/master/images/python.gif" weight="380px" height="270px" /></span>
  <span align="right"><img src="https://raw.githubusercontent.com/graykode/commit-autosuggestions/master/images/javascript.gif" weight="380px" height="270px" /></span>
</div>

### Abstract
[CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/pdf/2002.08155.pdf) introduces a pre-trained model in a combination of Program Language and Natural Language(PL-NL). It also introduces the problem of converting code into natural language (Code Documentation Generation).
```text
diff --git a/test.py b/test.py
new file mode 100644
index 0000000..1b1b82a
--- /dev/null
+++ b/test.py
@@ -0,0 +1,3 @@
+
+def add(a, b):
+    return a + b
```
```text
Recommended Commit Message : Add two arguments .
```
We can use CodeBERT to create a model that generates a commit message when code is added. However, most code changes are not made only by add of the code, and some parts of the code are deleted.
```text
diff --git a/test.py b/test.py
index 1b1b82a..32a93f1 100644
--- a/test.py
+++ b/test.py
@@ -1,3 +1,5 @@
+import torch
+import arguments
 
 def add(a, b):
     return a + b
```
```text
Recommended Commit Message : Remove unused imports
```
To solve this problem, use a new embedding called [`patch_type_embeddings`](https://github.com/graykode/commit-autosuggestions/blob/master/commit/model/diff_roberta.py#L40) that can distinguish added and deleted, just as the XLM(Lample et al, 2019) used language embeddeding. (1 for added, 2 for deleted.)

### Language support
| Language       | Added | Diff |  Data(Only Diff) | Weights |
| :------------- | :---: | :---:| :---: | :---:|
| Python         | ✅    | ✅   | [423k](https://drive.google.com/drive/folders/1_8lQmzTH95Nc-4MKd1RP3x4BVc8tBA6W?usp=sharing) |  [Link](https://drive.google.com/drive/folders/1OwM7_FiLiwVJAhAanBPWtPw3Hz3Dszbh?usp=sharing)  |
| JavaScript     | ✅    | ✅   | [514k](https://drive.google.com/drive/folders/1-Hv0VZWSAGqs-ewNT6NhLKEqDH2oa1az?usp=sharing) |  [Link](https://drive.google.com/drive/folders/1Jw8vXfxUXsfElga_Gi6e7Uhfc_HlmOuD?usp=sharing)  |
| Go             | ⬜    | ⬜   | ⬜ |  ⬜  |
| JAVA           | ⬜    | ⬜   | ⬜ |  ⬜  |
| Ruby           | ⬜    | ⬜   | ⬜ |  ⬜  |
| PHP            | ⬜    | ⬜   | ⬜ |  ⬜  |
* ✅ — Supported
* ⬜ - N/A ️

We plan to slowly conquer languages that are not currently supported. However, I also need to use expensive GPU instances of AWS or GCP to train about the above languages. Please do a simple sponsor for this! Add data is [CodeSearchNet dataset](https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h).

### Quick Start
To run this project, you need a flask-based inference server (GPU) and a client (commit module). If you don't have a GPU, don't worry, you can use it through Google Colab.

#### 1. Run flask pytorch server.
Prepare Docker and Nvidia-docker before running the server.

##### 1-a. If you have GPU machine.
Serve flask server with Nvidia Docker. Check the docker tag for programming language in [here](https://hub.docker.com/repository/registry-1.docker.io/graykode/commit-autosuggestions/tags).
| Language       | Tag   |
| :------------- | :---: |
| Python         | py    |
| JavaScript     | js    |
| Go             | go    |
| JAVA           | java  |
| Ruby           | ruby  |
| PHP            | php   |

```shell script
$ docker run -it -d --gpus 0 -p 5000:5000 graykode/commit-autosuggestions:{language}
```

##### 1-b. If you don't have GPU machine.
Even if you don't have a GPU, you can still serve the flask server by using the ngrok setting in [commit_autosuggestions.ipynb](commit_autosuggestions.ipynb).

#### 2. Start commit autosuggestion with Python client module named `commit`.
First, install the package through pip.
```shell script
$ pip install commit
```

Set the endpoint for the flask server configured in step 1 through the commit configure command. (For example, if the endpoint is http://127.0.0.1:5000, set it as follows: `commit configure --endpoint http://127.0.0.1:5000`)
```shell script
$ commit configure --help       
Usage: commit configure [OPTIONS]

Options:
  --profile TEXT   unique name for managing each independent settings
  --endpoint TEXT  endpoint address accessible to the server (example :
                   http://127.0.0.1:5000/)  [required]

  --help           Show this message and exit.
```

All setup is done! Now, you can get a commit message from the AI with the command commit.

```shell script
$ commit --help          
Usage: commit [OPTIONS] COMMAND [ARGS]...

Options:
  --profile TEXT       unique name for managing each independent settings
  -f, --file FILENAME  patch file containing git diff (e.g. file created by
                       `git add` and `git diff --cached > test.diff`)

  -v, --verbose        print suggested commit message more detail.
  -a, --autocommit     automatically commit without asking if you want to
                       commit

  --help               Show this message and exit.

Commands:
  configure
```

### Training detail
Refer [How to train for your lint style](docs/training.md). This allows you to re-fine tuning to your repository's commit lint style.

### Contribution
You can contribute anything, even a typo or code in the article. Don't hesitate!!.
Versions are managed only within the branch with the name of each version. After being released on Pypi, it is merged into the master branch and new development proceeds in the upgraded version branch.

### Author
[Tae Hwan Jung(@graykode)](https://github.com/graykode)
