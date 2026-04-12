import os.path as op

import pytest
import pyActigraphy
import inspect
from docstring_parser import parse

from pyActigraphy.utils.register import register, REGISTER

FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
awd_path = op.join(data_dir, 'test_sample.AWD')

# read AWD with default parameters
rawAWD = pyActigraphy.io.read_raw_awd(awd_path)


def test_register():
    # testing that REGISTER contains functions
    assert len(REGISTER) > 0, "REGISTER is empty"

    for cat, functs in REGISTER.items():
        assert len(functs) > 0, "{} is empty".format(cat)
        for func in functs:
            assert callable(func), "{} not all elements are callable"

    # Testing inclusion of function into REGISTER
    @register("TEST")
    def to_test():
        return True

    assert REGISTER.get("TEST")
    func = REGISTER["TEST"].pop(-1)
    assert func is to_test, "to_test wasn't included into REGISTER"

    if len(REGISTER["TEST"]) == 0:
        REGISTER.pop("TEST")


def test_belonging():
    for cat in REGISTER:
        for f in REGISTER[cat]:
            name = f.__name__
            assert name in dir(rawAWD), \
                "{}/{} not defined for AWD".format(cat, name)


def test_doc_extraction():
    for cat in REGISTER:
        for f in REGISTER[cat]:
            name = f.__name__
            assert callable(f), "{}/{} not callable".format(cat, name)
            sig = inspect.signature(f)
            try:
                doc = parse(f.__doc__)
            except Exception as err:
                pytest.fail("{}/{} failed parse doc for {}"
                            .format(cat, name, err))

            # Testing elements
            assert len(doc.short_description) > 0, \
                "{}/{} missing short description".format(cat, name)
            assert len(doc.long_description) > 0, \
                "{}/{} missing long description".format(cat, name)

            # Testing parameters
            nargs = len(sig.parameters)
            assert nargs > 0, "{}/{} dont't have any argumets"
            assert list(sig.parameters.keys())[0] == "self", \
                "{}/{} first argument not 'self'".format(cat, name)
            assert (nargs - 1) == len(doc.params), \
                "{}/{} not all parameters are described"

            for arg in doc.params:
                arg_name = arg.arg_name
                assert arg_name in sig.parameters, \
                    "{}/{} '{}' not in singnature".format(cat, name, arg_name)

                param = sig.parameters[arg_name]
                # Checking if default defined
                assert (param.default != inspect._empty) == arg.is_optional, \
                    "{}/{} Default status of '{}' do not match documentation"\
                    .format(cat, name, arg_name)

                # Reliable type checks can be done only with hints

                # Checking if defult value is reported
                if arg.is_optional:
                    assert arg.default is not None, \
                        "{}/{} Default value of '{}' not reported"\
                        .format(cat, name, arg_name)

                    # docstring_parser reports string defaults incorectly
                    if arg.type_name == "str":
                        continue
                    assert str(param.default) == arg.default, \
                        "{}/{} Default value of '{}' reported incorrectly, "\
                        "got {}, expected {}"\
                        .format(cat, name, arg_name,
                                str(param.default), arg.default)
