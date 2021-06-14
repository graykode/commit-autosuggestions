
import argparse
import jsonlines
from gitparser import *
from tree_sitter import Language, Parser

def main(args):
    PARSER = Parser()
    PARSER.set_language(Language(args.tree_sitter, 'java'))

    for (f_diff, f_msg) in [
        ('train.26208.diff', 'train.26208.msg'),
        ('valid.3000.diff', 'valid.3000.msg'),
        ('test.3000.diff', 'test.3000.msg')
    ]:
        name = f_diff.split('.')[0]
        with open(f_diff) as f:
            diff = f.readlines()
        with open(f_msg) as f:
            msg = f.readlines()

        for i, original_line in enumerate(diff):
            line = original_line.strip()
            added, deleted = [], []

            cleaned_message = message_cleaner(msg[i])
            if not cleaned_message:
                continue
            commit_tokens = tokenize_docstring_from_string(cleaned_message)

            if line.split()[0] != 'mmm':
                continue
            if line.split('<nl>')[0].split()[-1] != "java" or line.split('<nl>')[1].split()[-1] != "java":
                continue
            lines = [l.strip() for l in line.split('<nl>')[2:]]

            if args.nngen:
                with open(f"nngen/{name}.diff", mode="a") as f:
                    f.write(original_line)
                with open(f"nngen/{name}.msg", mode="a") as f:
                    f.write(msg[i])
                continue

            if args.adddel:
                for line in lines:
                    if len(line) > 0 and line[0] == "+":
                        added += tokenize_code(PARSER, line[1:])
                for line in lines:
                    if len(line) > 0 and line[0] == "-":
                        deleted += tokenize_code(PARSER, line[1:])

                added = ' '.join(added).replace('\n', ' ')
                added = ' '.join([v for v in added.strip().split() if v != '<nl>'])

                deleted = ' '.join(deleted).replace('\n', ' ')
                deleted = ' '.join([v for v in deleted.strip().split() if v != '<nl>'])

                with jsonlines.open(f"adddel/{name}.jsonl", mode="a") as writer:
                    writer.write(
                        {
                            "commit_tokens": commit_tokens,
                            "add_tokens": added,
                            "del_tokens": deleted,
                        }
                    )
            else:
                for line in lines:
                    added += [v for v in tokenize_code(PARSER, line) if v != '<nl>']

                with jsonlines.open(f"all/{name}.jsonl", mode="a") as writer:
                    writer.write(
                        {
                            "docstring_tokens": commit_tokens,
                            "code_tokens": added,
                        }
                    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--adddel", action='store_true')
    parser.add_argument("--nngen", action='store_true')
    parser.add_argument("--tree_sitter", type=str, required=True,
                        help="The path of py-tree-sitter-languages.so (ex, /src/build/py-tree-sitter-languages.so)")
    args = parser.parse_args()

    main(args)