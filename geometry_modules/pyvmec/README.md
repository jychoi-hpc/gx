A module that can calculate the geometry coefficients for both tokamaks and stellarators from a VMEC wout file.

Additionally, this module can calculate the geometry coefficients when the user varies tprim(a\_N/L\_T), fprim(a\_N/L\_n), beta or global shear for a given equilibrium according to the formalism of [Greene and Chance](https://iopscience.iop.org/article/10.1088/0029-5515/21/4/002) for tokamaks, and [Hegna and Nakajima](https://pubs.aip.org/aip/pop/article-abstract/5/5/1336/104335/On-the-stability-of-Mercier-and-ballooning-modes?redirectedFrom=fulltext) for stellarators. This could have significant implications for a transport solver such as Tr3D as we currently do not change geometric coefficients when we calculate the heat/particle flux jacobian with respect to fprim and tprim.

This module does not suffer from the pathologies of a generalized local 3D equilibrium (as derived by Boozer and Parra Diaz). The source of the problem in a 3D local Miller-like equilibrium, which is the constraint J dot grad psi = 0, is satisfied by default VMEC equilibrium.

All the calculations are performed in Boozer coordinates as the geometric quantities are more intuitive, making the analysis relatively straightforward.

Dependencies: booz\_xform, netcdf4. Both these packages can be installed using pip.

If you use the s-alpha 3D geometry functionality for your work, please cite the [following paper](https://arxiv.org/abs/2410.04576) 

## To dos:

* Currently, I have only compared the coefficients with GS2 for tokamak equilibria. I have to add tests to avoid sign-related issues described in PR #25

* The 3D geometry variation still needs testing, but to do that, I need access to a code that has done 3D variation before. The only code I remember is STESA (by Hudson), but it is not publicly available.


