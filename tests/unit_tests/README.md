README for notes on using Catch2

To compile when in the catch2_tests directory:
'cd build'
'cmake ..'
'make'

- Right now the source code and tests are in the same .cpp file, and the catch.hpp header is in the same directory, so the compiling could also just be done trivially on the command line
- All that is required to use catch2 is #include "catch.hpp" in the test.cpp file
- In pure test files, it isn't necessary to have a main(), assuming that the macro #define CATCH_CONFIG_MAIN is added at the top of the test file

------------------------
- The function fizzbuzz(int) is a toy problem that returns "fizz" if the integer is divisible by 3, "buzz" if divisible by 5, "fizzbuzz" if divisible by both 3 and 5, and the number itself if not divisble by either.
- I'll use this to do some very simple catch2 tests

-------------------------
- The "TEST_CASE" keyword simply sets up some tests to perform. It can be provided with some naming string and an optional tag

- In cpp this takes the form:
TEST_CASE("test name", "[tag_name]") {
		do stuff
		}

- In each test case one can make assertions
- If the "REQUIRE" assertion fails, the whole TEST_CASE fails and no further lines in that TEST_CASE are read
- If the "CHECK" assertion fails, only that assertion fails, and the remaining lines in that TEST_CASE will still be executed
- Using the fizzbuzz function with "0" as an argument should return "0" since it isn't divisible by 3 or 5, so a test could look like this:

TEST_CASE("check zero", "[tag]") {
		 REQUIRE(fizzbuzz(0) == "0"); // pass
		 CHECK(fizzbuzz(0) == "1"); // will fail, but next line is still read
		 REQUIRE(fizzbuzz(0) == "1"); // fail, next line not read
		 std::cout << "finished\n"; // doesn't get printed
		 }

- It is also possible to have SECTIONS in each TEST_CASE, which are each executed from the start (i.e. local changes to variables in one SECTION will not be passed to another)

- If making assertions with floating points, it is possible to give a tolerance or margin for how close the result must be to the reference (see the final TEST_CASE in test.cpp)

------------------------------------------------------------

- To run all the tests, simply give ./unit_tests
- If you only want to run tests with certain tags, give ./unit_tests [tag_name]