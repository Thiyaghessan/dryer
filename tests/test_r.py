from pathlib import Path

from dryer.discover import is_test_file, iter_source_files, language_of
from dryer.extract import entries_in_source
from dryer.scan import find_duplicates, scan_files
from dryer.shape import jaccard


def write_source(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def score(left: str, right: str, language: str = "r", suffix: str = "R") -> float:
    """Similarity of the first function in `left` and the first function in `right`."""

    left_entries, _ = entries_in_source(language, left, f"left.{suffix}", f"left.{suffix}")
    right_entries, _ = entries_in_source(language, right, f"right.{suffix}", f"right.{suffix}")
    return jaccard(left_entries[0].fingerprints, right_entries[0].fingerprints)


R_LEFT = """\
alpha <- function(xs, limit = 10) {
  ys <- dplyr::filter(xs, amount > limit)
  if (nrow(ys) == 0L && !is.null(xs)) {
    return(NULL)
  }
  return(purrr::map(ys$amount, round, digits = 2))
}
"""

R_RENAMED = """\
beta = function(items, cap = 99) {
  kept <- dplyr::filter(items, total > cap)
  if (nrow(kept) == 5L && !is.null(items)) {
    return(NULL)
  }
  return(purrr::map(kept$total, round, places = 7))
}
"""


def test_renamed_locals_and_changed_literals_score_one(tmp_path):
    write_source(tmp_path, "left.R", R_LEFT)
    write_source(tmp_path, "right.R", R_RENAMED)
    files = sorted(tmp_path.glob("*.R"))
    entries, warnings = scan_files(files, tmp_path, min_lines=1, min_nodes=1)
    assert warnings == []
    found = find_duplicates(entries, threshold=0.99)
    assert [(item.language, item.left.file, item.right.file, item.score) for item in found] == [
        ("r", "left.R", "right.R", 1.0)
    ]
    assert (found[0].left.start_line, found[0].left.end_line) == (1, 7)


def test_a_different_called_function_lowers_the_score():
    assert score(R_LEFT, R_RENAMED.replace("dplyr::filter", "dplyr::arrange")) < 1.0
    assert score(R_LEFT, R_RENAMED.replace("nrow(", "ncol(")) < 1.0


def test_the_package_in_front_of_a_call_is_part_of_its_name():
    left = "f <- function(x) {\n  dplyr::filter(x)\n}\n"
    assert score(left, "g <- function(y) {\n  dplyr::filter(y)\n}\n") == 1.0
    assert score(left, "g <- function(y) {\n  stats::filter(y)\n}\n") < 1.0


def test_a_different_operator_lowers_the_score():
    left = "f <- function(a, b) {\n  a %in% b\n}\n"
    assert score(left, "g <- function(x, y) {\n  x %in% y\n}\n") == 1.0
    assert score(left, "g <- function(x, y) {\n  x %>% y\n}\n") < 1.0
    assert score(left, "g <- function(x, y) {\n  x <- y\n}\n") < 1.0
    assert score(left, "g <- function(x, y) {\n  x:y\n}\n") < 1.0
    assert score(R_LEFT, R_RENAMED.replace("total > cap", "total < cap")) < 1.0
    assert score(R_LEFT, R_RENAMED.replace("&&", "||")) < 1.0


def test_every_kind_of_literal_is_the_same_token():
    left = "f <- function() {\n  c(1, 2L, 'a', TRUE, NULL, NA, Inf, NaN, 3i)\n}\n"
    right = "g <- function() {\n  c('b', NA, 7, NULL, FALSE, 4i, NaN, 9L, Inf)\n}\n"
    assert score(left, right) == 1.0


def test_a_called_member_keeps_its_name_and_a_read_member_does_not():
    left = "f <- function(x) {\n  x$total\n}\n"
    assert score(left, "g <- function(y) {\n  y$count\n}\n") == 1.0
    called = "f <- function(x) {\n  x$open()\n}\n"
    assert score(called, "g <- function(y) {\n  y$open()\n}\n") == 1.0
    assert score(called, "g <- function(y) {\n  y$close()\n}\n") < 1.0


def test_comments_do_not_change_the_fingerprint():
    commented = R_RENAMED.replace("  kept <-", "  # keep the large ones\n  kept <-")
    assert score(R_LEFT, commented) == 1.0


def test_only_named_functions_outside_other_functions_are_compared():
    source = """\
outer <- function(xs) {
  inner <- function(x) {
    x + 1
  }
  purrr::map(xs, function(x) inner(x))
}
handlers <- list(on_error = function(e) NULL)
answer <- 42
"""
    entries, warning = entries_in_source("r", source, "a.R", "a.R")
    assert warning is None
    assert [(entry.start_line, entry.end_line) for entry in entries] == [(1, 6)]


def test_the_score_means_the_same_thing_as_in_python():
    """The same change gives the same verdict in both languages."""

    python_left = "def alpha(xs):\n    ys = keep(xs, odd)\n    return apply(ys, 1)\n"
    python_renamed = "def beta(items):\n    kept = keep(items, even)\n    return apply(kept, 2)\n"
    python_other = "def beta(items):\n    kept = drop(items, even)\n    return apply(kept, 2)\n"
    r_left = "alpha <- function(xs) {\n  ys <- keep(xs, odd)\n  return(apply(ys, 1))\n}\n"
    r_renamed = "beta <- function(items) {\n  kept <- keep(items, even)\n  return(apply(kept, 2))\n}\n"
    r_other = "beta <- function(items) {\n  kept <- drop(items, even)\n  return(apply(kept, 2))\n}\n"

    assert score(python_left, python_renamed, "python", "py") == 1.0
    assert score(r_left, r_renamed) == 1.0
    assert 0.0 < score(python_left, python_other, "python", "py") < 1.0
    assert 0.0 < score(r_left, r_other) < 1.0


def test_r_files_are_found_and_test_files_are_left_out(tmp_path):
    assert language_of("R/build.R") == "r"
    assert language_of("R/build.r") == "r"
    assert is_test_file("test_build.R") is True
    assert is_test_file("test-build.r") is True
    assert is_test_file("R/build.R") is False
    write_source(tmp_path, "R/build.R", "f <- function() 1\n")
    write_source(tmp_path, "R/test-build.R", "f()\n")
    write_source(tmp_path, "tests/testthat/test-build.R", "f()\n")
    found = iter_source_files([tmp_path])
    assert [path.relative_to(tmp_path.resolve()).as_posix() for path in found] == ["R/build.R"]


def test_equals_assignment_is_the_same_as_arrow_assignment():
    arrow = "f <- function(x) {\n  y <- x\n  y\n}\n"
    equals = "g = function(a) {\n  b = a\n  b\n}\n"
    assert score(arrow, equals) == 1.0
    # A named argument's `=` is not an assignment and stays what it is.
    named = "f <- function(x) {\n  round(digits = x)\n}\n"
    assigned = "f <- function(x) {\n  round(digits <- x)\n}\n"
    assert score(named, assigned) < 1.0


def test_how_a_name_is_reached_is_part_of_the_shape():
    exported = "f <- function(x) {\n  p::g(x)\n}\n"
    assert score(exported, "h <- function(y) {\n  p::g(y)\n}\n") == 1.0
    # `:::` reaches a function the package does not export.
    assert score(exported, "h <- function(y) {\n  p:::g(y)\n}\n") < 1.0
    # The same holds when the name is read, not called.
    assert score("f <- function() {\n  p::g\n}\n", "h <- function() {\n  p:::g\n}\n") < 1.0

    list_member = "f <- function(x) {\n  x$a\n}\n"
    assert score(list_member, "h <- function(y) {\n  y$b\n}\n") == 1.0
    # `@` reads a slot of an S4 object, `$` a member of a list.
    assert score(list_member, "h <- function(y) {\n  y@b\n}\n") < 1.0
    assert score("f <- function(x) {\n  x$run()\n}\n", "h <- function(y) {\n  y@run()\n}\n") < 1.0


def test_a_function_given_a_quoted_name_is_compared():
    source = '"quoted" <- function(x) {\n  x + 1\n}\n`back tick` <- function(x) {\n  x + 1\n}\n'
    entries, _warning = entries_in_source("r", source, "a.R", "a.R")
    assert [(entry.start_line, entry.end_line) for entry in entries] == [(1, 3), (4, 6)]


def test_a_quoted_name_is_a_name_not_a_literal():
    plain = "f <- function(x) {\n  round(x, digits = 2)\n}\n"
    quoted = '"g" <- function(y) {\n  round(y, "digits" = 3)\n}\n'
    assert score(plain, quoted) == 1.0
    # A string that is a value is still a literal.
    assert score(plain, "f <- function(x) {\n  round(x, digits = 'a')\n}\n") == 1.0
    assert score(plain, "f <- function(x) {\n  round('x', digits = 2)\n}\n") < 1.0
