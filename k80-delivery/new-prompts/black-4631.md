# Trailing comments disappear when the first comment is marked fmt: skip

Formatting a file with a trailing comment block can fail when the first comment line includes `# fmt: skip`; additional comment lines are removed and Black reports an internal formatting error.

Reproduce with:

```python
def foo():
    pass


# comment1  # fmt: skip
# comment2
```

Run `black file.py`. Black should retain the comments and finish without an internal error.
