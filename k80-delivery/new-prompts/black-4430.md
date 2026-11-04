# Line-range formatting changes code outside the requested range

Black's `--line-ranges` option is expected to limit formatting to the requested lines, but formatting a selected section also changes whitespace and code outside that section.

Reproduce with this exact file:

```python
print("format me"  )
 # format whitespace
  # format whitespace
   # format whitespace
print( "format me" )
   # format whitespace
  # format whitespace
 # format whitespace
print(  "format me")



print("don't format me"  )
 # don't format whitespace
  # don't format whitespace
   # don't format whitespace
print( "don't format me" )
   # don't format whitespace
  # don't format whitespace
 # don't format whitespace
print(  "don't format me")
```

Run:

```sh
black --line-ranges 1-9 file.py
```

The report shows Black normalizing comments in the second block too. Only lines 1-9 are selected; the indentation and three-line separation in the later block should remain as written.
