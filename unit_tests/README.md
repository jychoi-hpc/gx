NOTE: most unit tests are old and broken, and need to be updated to run with latest version of the code. this is a work in progress...

To make unit tests, first must execute (only on first time)
```
git submodule init
git submodule update
```
to initialize googletests.

Then the unit tests can to built with
```
make -j
```
and ran with 
```
./unit_tests
```
