"""Pre-match feature engineering.

Every feature produced here must be computable strictly from information
available *before* kickoff. Rolling/scorer/player features use ``shift``-style
logic and as-of merges to guarantee this.
"""
