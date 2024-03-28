# VMEC to GX Geometry Interface

This module generates the geometric coeffcients required for a local, non-linear run with the GX code. The geometric coefficients are calcuated on an equispaced, equal-arc, field-line-following coordinate. After this, they are saved in the form of a data file in both grid and netCDF file formats.

After running the module and obtaining the GX equilibrium data file, you must set igeo=1 and geofile=path/to/the/equilibrium-data-file in the Geometry section of your GX input file.


## Input Parameters
There are a few input parameters that you need to choose before generating an equilibrium data file. Typical values of these parameters can be found in the sample\_inpfile2.ing file in the geometry\_modules/vmec/input\_files directory. The meaning of different options is explained below: 

- alpha: Magnetic field line label. alpha=0.0 would correspond to a flux tube with the center at the outboard midplane and the center of one of the symmetric field periods.
- nzgrid: The number of grid points in GX will be 2\*nzgrid+1
- npol: Sets limits of the flux tube to be [-npol\*pi, npol\*pi]
- desired\_normalized\_toroidal\_flux :lies in the range [0, 1] with 0 being the magnetic axis and 1, the boundary
- vmec\_surface\_option:

	* 0 - interpolates quantities between VMEC's half and full grid to get geometric quantities at exactly the "desired\_normalized\_toroidal\_flux" input
	* 1 - calculates quantities on the closest surface of VMEC's half grid to "desired\_normalized\_toroidal\_flux"
	* 2 - calculates quantities on the closest surface of VMEC's full grid to "desired\_normalized\_toroidal\_flux"

### Choosing custom flux tube sizes (For experts only)

If desired, one can choose the exact endpoints of the flux tube, or have the endpoints coincide with the zeros of gds21 or gbdrift0. The input parameters to control this feature are below:

* flux\_tube\_cut:

	* "none" - This is the default parameter corresponding to no cutting. npol will completely control the length
	* "custom" - Choose the exact endpoints of the tube. This option is buggy right now.
	* "gds21" - Choose a particular zero of the gds21 array as the endpoints
	* "gbdrift0" - Choose a particular zero of the gbdrift0 array as the endpoints

* custom\_length: If "custom" is chosen for flux\_tube\_cut, the theta values for the flux tube will be [-custom\_length , custom\_length]
* which\_crossing: If either "gds21" or "gbdrift0" are chosen for flux\_tube\_cut, this integer value will specify which zero to use, since these functions will have numerous crossing. Choosing "1" will yield the first zero from the center, "2" will yield the second, and so on.

## Running the module

After compiling the GX code you will find an executable ./geometry\_modules/vmec/convert\_VMEC\_to\_GX. Once you have written an input file (see ./geometry\_modules/vmec/input\_files/sample\_inpfile1.ing) you can either run 
```
/path/to/convert_VMEC_to_GX path/to/input_file

```
The above command will work if you have added the absolute paths of the input file and the vmec file in the input file. Otherwise, you can place the input and vmec files in the same directory (see ./geometry\_modules/vmec/input\_files/sample\_inpfile2.ing) and upon moving to that directory, run

```
/path/to/convert_VMEC_to_GX ./input_file

```
## Test Cases

The tests/ directory contains 3 separate VMEC equilibria, as well as the corresponding outputs from the GIST code, which similarly computes geometric quantities for GENE and GS2 from a VMEC input.
You can choose any of the 3 equilibria with some combination of input paramters and compare the results from this module with GIST

## Acknowledgements

This code, written by Dr. Mike Martin, is a translation of Dr. Matt Landreman's full\_surface\_gs2\_vmec\_interface.


