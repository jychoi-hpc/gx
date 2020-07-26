# GX Unit Tests

## Prerequisites

- All that is required to use catch2 is the header "catch.h", with the appropriate #include "catch.h" in the source files

## Compiling

Compiling the regression and unit tests requires the user to give, in the main directory:

```
make unit_tests
```

## Running Regression and/or Unit Tests

All tests are written using the Catch2 header library (documentation can be found at: <a href="https://github.com/catchorg/Catch2">Catch2</a>). To run all tests, in the main directory give:

```
./unit_tests
```

However, the tests are set up in such a way that allows the user to run only a subset of all these tests through the use of a Catch2 feature known as a tag. This can be done by simply adding the tag (details for how to find the tag are given below) as an argument to the previous command:

```
./unit_tests [tag_name]
```

## Test Sources

The source files for the tests can be found in GX/tests. Specifically, regression tests are located in GX/tests/regression/regression.cpp, and unit tests in GX/tests/unit_tests/initial_tests.cpp. The aforementioned "tags" can be identified by the third argument to the TEST_CASE_METHOD function in the source files above.

## Some Notes on Catch2

- An example source file showing basic Catch2 functionality can be found in GX/tests/unit_tests/basic_examples/test_examples.cpp
- The "TEST_CASE" keyword simply sets up some tests to perform. It can be provided with some naming string and an optional tag
- In each test case one can make assertions
- If the "REQUIRE" assertion fails, the whole TEST_CASE fails and no further lines in that TEST_CASE are read
- If the "CHECK" assertion fails, only that assertion fails, and the remaining lines in that TEST_CASE will still be executed
- It is also possible to have SECTIONS in each TEST_CASE, which are each executed from the start (i.e. local changes to variables in one SECTION will not be passed to another)
- If making assertions with floating points, it is possible to give a tolerance or margin for how close the result must be to the reference (see the final TEST_CASE in test_examples.cpp)