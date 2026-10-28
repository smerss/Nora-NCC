# Successive prefix relation

A sequence can be transformed into a sequence of its successive non-empty prefixes.
The first output contains only the first input item. Each later output preserves the previous output and includes the next input item in the original order.
This describes a relation between input and output, not a library function or an implementation.

The following are data examples for learning the relation, not Python code:

input: "abca"
outputs: "a" | "ab" | "abc" | "abca"

input: "xy"
outputs: "x" | "xy"
