# A short line length makes a valid `case case` pattern fail to parse

Black can format a `case case` pattern at line length 14, but reports a parse error when the line length is reduced to 13 or less.

Reproduce with:

```python
match test:
    case case if True:
        pass
```

Run Black with a line length of 13 and then 14. The valid pattern should remain parseable at both settings.
