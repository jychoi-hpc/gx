inline void ExBshear(cuComplex *Phi, cuComplex **Dens, cuComplex **Upar, cuComplex **Tpar,
         cuComplex **Tprp, cuComplex **Qpar, cuComplex **Qprp,
	 float* kx_shift, int* jump, float avgdt)
{
    // shift moments and fields in kx to account for ExB shear
    kxshift<<<dimGrid,dimBlock>>>(kx_shift,jump,ky,g_exb,avgdt);
    shiftField<<<dimGrid,dimBlock>>>(Phi,jump);
    for(int s=0; s<nSpecies; s++) {
      shiftField<<<dimGrid,dimBlock>>>(Dens[s],jump);
      shiftField<<<dimGrid,dimBlock>>>(Upar[s],jump);
      shiftField<<<dimGrid,dimBlock>>>(Tpar[s],jump);
      shiftField<<<dimGrid,dimBlock>>>(Tprp[s],jump);
      shiftField<<<dimGrid,dimBlock>>>(Qpar[s],jump);
      shiftField<<<dimGrid,dimBlock>>>(Qprp[s],jump);
    }
}
