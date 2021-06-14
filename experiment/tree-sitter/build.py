
from tree_sitter import Language

languages = [
    'tree-sitter-python',
    'tree-sitter-javascript',
    'tree-sitter-go',
    'tree-sitter-ruby',
    'tree-sitter-java',
    'tree-sitter-php',
]

# See how to build : https://github.com/tree-sitter/py-tree-sitter

Language.build_library(
    # Store the library in the directory
    'build/py-tree-sitter-languages.so',
    # Include one or more languages
    languages
)

