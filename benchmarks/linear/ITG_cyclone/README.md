These tests run linear ion-temperature-gradient (ITG) instability calculations.
`itg_salpha_adiabatic_electrons.in` uses a circular s-alpha geometry with Cyclone-base-case parameters and a Boltzmann adiabatic electron response.
`itg_miller_adiabatic_electrons.in` uses a circular Miller geometry with Cyclone-base-case parameters and a Boltzmann adiabatic electron response.
`itg_miller_kinetic_electrons.in` uses a circular Miller geometry with Cyclone-base-case parameters and kinetic electrons.

To run a test, simply use (taking `itg_miller_adiabatic_electrons` as an example)
```
[/path/to/]gx itg_miller_adiabatic_electrons.in
```

To check the results, use
```
python check.py itg_miller_adiabatic_electrons
```

