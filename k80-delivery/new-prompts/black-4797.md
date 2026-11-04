# Python 3.14 template strings are rejected

When asked to target Python 3.14, Black rejects a file containing a template string as invalid syntax.

Reproduce with:

```python
name = "world"
print(t"hello {name}")
```

```sh
black file.py --target-version py314
```

Black reports a parse error at the template string. This valid Python 3.14 syntax should be accepted for formatting.
