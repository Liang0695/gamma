# Multiline f-string with a multiline string in its expression cannot be parsed

Black fails to parse a Python file containing a multiline f-string whose replacement expression contains a multiline string literal.

Reproduce by formatting such a file with Black. Black should accept the valid syntax and complete formatting without a parse error.
