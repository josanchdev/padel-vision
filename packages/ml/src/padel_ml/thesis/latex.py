"""Numbers as the dissertation prints them (memoria/GUIA.md, sections 2 and 7).

One decimal for percentages, three for F1 and other proportions. Values are
written with a decimal point and siunitx prints the comma and the spacing that
Spanish typography asks for («82,7 %», «77 300»).

Rounding is half-up, as done by hand, from the shortest repr of the float:
Python's round() goes half to even, and float arithmetic adds noise
(0.8345 * 100 == 83.44999...), so either would print 83,4 % where 83,5 % is due.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

PERCENT_PLACES = 1
PROPORTION_PLACES = 3


def fixed(value: float, places: int, *, scale: int = 1) -> str:
    """`value * scale` rounded half-up to `places` decimals, with a decimal point."""
    exact = Decimal(repr(float(value))) * scale
    rounded = exact.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return str(rounded + 0)  # "+ 0" turns a negative zero into zero


def percent(share: float) -> str:
    """A proportion (0-1) as a percentage: 0.8271 -> 82,7 %."""
    return rf"\SI{{{fixed(share, PERCENT_PLACES, scale=100)}}}{{\percent}}"


def points(share: float) -> str:
    """A difference of proportions in percentage points: 0.0376 -> 3,8."""
    return rf"\num{{{fixed(share, PERCENT_PLACES, scale=100)}}}"


def proportion(value: float, places: int = PROPORTION_PLACES) -> str:
    """F1, precision, recall and other 0-1 values: 0.95816 -> 0,958."""
    return rf"\num{{{fixed(value, places)}}}"


def number(value: float, places: int) -> str:
    """Any other decimal number."""
    return rf"\num{{{fixed(value, places)}}}"


def count(value: int) -> str:
    """A whole count: 2377 -> 2377, 77300 -> 77 300."""
    return rf"\num{{{int(value)}}}"


def quantity(value: float, places: int, unit: str) -> str:
    """A number with its unit, separated as the RAE asks: 67.46 s -> 67,5 s."""
    return rf"\SI{{{fixed(value, places)}}}{{{unit}}}"


def percent_range(low: float, high: float) -> str:
    """A 95 % interval of a proportion, in percent: 80,3-85,0."""
    low_text = fixed(low, PERCENT_PLACES, scale=100)
    high_text = fixed(high, PERCENT_PLACES, scale=100)
    return rf"\num{{{low_text}}}--\num{{{high_text}}}"


def proportion_range(low: float, high: float) -> str:
    """A 95 % interval of an F1 or similar: 0,939-0,955."""
    return rf"\num{{{fixed(low, PROPORTION_PLACES)}}}--\num{{{fixed(high, PROPORTION_PLACES)}}}"


def tabular(
    spec: str, header: list[str], rows: list[list[str] | str], *, rules: list[int] | None = None
) -> str:
    """A booktabs tabular: header in bold, no vertical rules, a \\midrule after the
    header and before each row index in `rules`. A row given as a plain string is
    a group heading that spans the whole width, in italics."""
    width = len(header)
    for row in rows:
        if isinstance(row, list) and len(row) != width:
            raise ValueError(f"row with {len(row)} cells in a {width}-column table: {row}")
    breaks = set(rules or [])
    body = []
    for index, row in enumerate(rows):
        if index in breaks:
            body.append("\\midrule")
        if isinstance(row, str):
            body.append(f"\\multicolumn{{{width}}}{{@{{}}l}}{{\\emph{{{row}}}}} \\\\")
        else:
            body.append(" & ".join(row) + " \\\\")
    return "\n".join(
        [
            f"\\begin{{tabular}}{{{spec}}}",
            "\\toprule",
            " & ".join(f"\\textbf{{{cell}}}" for cell in header) + " \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
