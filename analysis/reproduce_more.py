"""Reproduce the numbers of the article on the price premium and on the prior variance from the files in ../data.

    python reproduce_more.py            runs the scripts in scripts/ one after the other and prints their output
    python reproduce_more.py --check    also compares every output with the file of the same name in expected_more/

Needs Python 3.10 or newer with numpy, pandas and scipy. The study files already hold the premium of the market
simulator and the true premium, so the Choicery app is not needed. The fit of the formula tries several starting
values and leaves out every design once, which takes a few minutes.

Script                      What it gives
test_new_sample_sizes.py    the formula on 768 studies with 50, 100 and 250 respondents that were not part of the fit
fit_formula.py              the formula of the price premium as the calculator uses it, fitted on all blocks: constants,
                            the 90% point, how many premiums are inside the margin by design, levels, agreement and
                            respondents, the same with a design left out of the fit, and outside the reference case
figures.py                  the numbers behind Figures 1 to 3 and the comparisons quoted with them
question_formats.py         dual-response and best-worst against CBC on the same people and tasks
prior_variance_shares.py    the error of a choice share with prior variance 4.0, 1.0 and 0.25 on identical answers
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["fit_formula.py", "test_new_sample_sizes.py", "figures.py", "question_formats.py", "prior_variance_shares.py"]


def main():
    expected = HERE / "expected_more"
    different = []
    for name in SCRIPTS:
        done = subprocess.run([sys.executable, str(HERE / "scripts" / name)], capture_output=True, encoding="utf-8", cwd=str(HERE / "scripts"))
        if done.returncode != 0:
            sys.exit(f"{name} failed:\n{done.stderr[-2000:]}")
        out = done.stdout.replace("\r\n", "\n")
        print(f"\n################ {name}\n{out}", end="")
        target = expected / (name[:-3] + ".txt")
        if "--write-expected" in sys.argv:
            expected.mkdir(exist_ok=True)
            target.write_text(out, encoding="utf-8", newline="\n")
        if "--check" in sys.argv and target.read_text(encoding="utf-8") != out:
            different.append(name)
    if "--check" in sys.argv:
        if different:
            sys.exit("output differs from expected_more/ for: " + ", ".join(different))
        print("\nall outputs match expected_more/")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
