"""Small samples (50, 100, 250) for the formula of 18.11.: the script of 17.11., unchanged, on the tuned-generator runs."""
import sys

import auswertung_neu as N  # sets up the redirected loaders and output folders

sys.modules["auswertung_gesamtfehler"] = N.G  # the old script imports this name
import auswertung_kleine_stichproben as K  # noqa: E402

K.PAR = dict(N.FORMULA)
K.FACTORS = {"share_of_preference": (1.0, 1.0), "rfc": (1.2, 1.17), "first_choice": (1.86, 1.77)}

if __name__ == "__main__":
    K.main()
