import sys
import pyActigraphy

from inspect import signature
from docstring_parser import parse

from pyActigraphy.register import REGISTER


raw = pyActigraphy.io.read_raw_mtn(sys.argv[1])

for cat in REGISTER:
    print(">>>>>", cat, ">>>>>>>>>>>>")
    for f in REGISTER[cat]:
        sig = signature(f)
        print("--", f.__name__, sig, "--")

        if f.__name__ not in dir(raw):
            print("W! Function is not defined for current raw object!")
            continue

        doc = parse(f.__doc__)
        print(doc.short_description)
        print(doc.long_description)
        for arg in doc.params:
            if arg.arg_name not in sig.parameters:
                print("W: Argument {} not in the list of params!"
                      .format(arg.arg_name))
                continue
            out = ("{}: {}".format(arg.arg_name, arg.type_name))
            if arg.is_optional:
                out += " (optional, {}/{})".format(
                        arg.default,
                        sig.parameters[arg.arg_name].default)
            print("\t", out)
            print("\t", arg.description.replace("\n", " "))
        res = f(raw)
        print("Result:", res)
        print("--------------------\n")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
