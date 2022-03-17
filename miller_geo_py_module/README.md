The python script gx_geo.py can be used to generate an eik.out file with geometry coefficients, which can be read by gx with igeo=1. 

To generate the geometry, specify the Miller parameters in your gx input file. For example, in an input file called gx.in, one might have
...
[Geometry]
 igeo = 1
 geofile = "gx.eik.out"
 rhoc = 0.5
 shat = 0.8
 Rmaj = 2.77778
 R_geo = 2.77778
 shift = 0.0
 qinp = 1.4
 akappa = 1.0
 akappri = 0.0
 tri = 0.0
 tripri = 0.0
 betaprim = 0.0
...

Next, use the gx_geo.py script via
> python gx_geo.py gx.in

This will generate a file gx.eik.out with the geometry coefficients.

Now you can run gx via
> [/path/to/]gx gx.in
