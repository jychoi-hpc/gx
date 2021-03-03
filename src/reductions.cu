#include "reductions.h"
#include <iostream>

// static member to hold `global' handle to cuTensor library
cutensorHandle_t Red::handle;
// Before any instances of Red have been constructed nobody has
// called init.
bool Red::isCuTensorInitialised = false; 

// catch-all reduction to a scalar, for any block of N contiguous floats
Red::Red(int N, int nspecies) : N_(N)
{
  version_red = 'N';
  Addwork = nullptr;    Maxwork = nullptr;    AddworkT = nullptr;
  
  extent['a'] = N_;
  extent['s'] = nspecies;
  for (auto mode : Amode) extent_A.push_back(extent[mode]); // incoming tensor assuming data is contiguous
  for (auto mode : Bmode) extent_B.push_back(extent[mode]); // target scalar output
  for (auto mode : Qmode) extent_Q.push_back(extent[mode]); // incoming tensor without abs value
  for (auto mode : Rmode) extent_R.push_back(extent[mode]); // target species scalar output

  //  for (auto mode : Amode) cout << Amode[mode] << '\n';
  
  if( !isCuTensorInitialised ) {
     // First time, do initialisation
     HANDLE_ERROR( cutensorInit(&handle));
  }

  HANDLE_ERROR( cutensorInitTensorDescriptor(&handle, &dA, nAmode, extent_A.data(), NULL, cfloat, CUTENSOR_OP_ABS));
  HANDLE_ERROR( cutensorInitTensorDescriptor(&handle, &dB, nBmode, extent_B.data(), NULL, cfloat, CUTENSOR_OP_ABS));
  HANDLE_ERROR( cutensorInitTensorDescriptor(&handle, &dQ, nQmode, extent_Q.data(), NULL, cfloat, CUTENSOR_OP_IDENTITY));
  HANDLE_ERROR( cutensorInitTensorDescriptor(&handle, &dR, nRmode, extent_R.data(), NULL, cfloat, CUTENSOR_OP_IDENTITY));
}

// Reduction methods for WSPECTRA integrations (G**2 -> G**2(kx), for example)
Red::Red(Grids *grids, std::vector<int> spectra) : grids_(grids), spectra_(spectra)
{
  version_red = 'W';
  Addwork = nullptr;    Maxwork = nullptr;    AddworkT = nullptr;

  int J;  J = spectra_.size();
  initialized.assign(J, 0);   desc.resize(J);   sAdd.assign(J, 0);  extents.resize(J);
  
  extent['y'] = grids_->Nyc;
  extent['x'] = grids_->Nx;
  extent['z'] = grids_->Nz;
  extent['l'] = grids_->Nl;
  extent['m'] = grids_->Nm;
  extent['s'] = grids_->Nspecies;;

  for (auto mode : Wmode) extent_W.push_back(extent[mode]);;

  // Build tensor descriptions for partial summations here
  for (int j = 0; j < J; j++) {
    if (spectra_[j] == 1) {
      for (auto mode : Modes[j]) extents[j].push_back(extent[mode]);

      if( !isCuTensorInitialised ) {
         // First time, do initialisation
         HANDLE_ERROR( cutensorInit(&handle));
      }

      HANDLE_ERROR( cutensorInitTensorDescriptor(&handle, &dW, nWmode, extent_W.data(),
						 NULL, cfloat, CUTENSOR_OP_ABS));
      HANDLE_ERROR( cutensorInitTensorDescriptor(&handle, &desc[j], Modes[j].size(),
						 extents[j].data(), NULL, cfloat, CUTENSOR_OP_ABS));
    }
  }
}

// Reduction methods for PSPECTRA integrations ( integrating (1-G0)Phi**2, for example)
Red::Red(Grids *grids, std::vector<int> spectra, bool potential) : grids_(grids), spectra_(spectra)
{  
  version_red = 'P';
  Addwork = nullptr;    Maxwork = nullptr;    AddworkT = nullptr;

  int J;  J = spectra_.size();
  initialized.assign(J, 0);   desc.resize(J);   sAdd.assign(J, 0);  extents.resize(J);

  // Unnecessary conditional, but included to generate an error if the routine is called in the wrong context
  if (potential) {
    extent['y'] = grids_->Nyc;
    extent['x'] = grids_->Nx;
    extent['z'] = grids_->Nz;
    extent['s'] = grids_->Nspecies;;
  } else {
    printf("Exiting from Red constructor because potential is false \n");
    exit(1);
  }
  for (auto mode : Pmode) extent_P.push_back(extent[mode]);
  for (int j = 0; j < J; j++) {
    if (spectra_[j] == 1) {
      for (auto mode : pModes[j]) extents[j].push_back(extent[mode]);
      
      if( !isCuTensorInitialised ) {
         // First time, do initialisation
         HANDLE_ERROR( cutensorInit(&handle));
      }

      HANDLE_ERROR( cutensorInitTensorDescriptor(&handle, &dP, nPmode, extent_P.data(), NULL, cfloat, CUTENSOR_OP_ABS));
      HANDLE_ERROR( cutensorInitTensorDescriptor(&handle, &desc[j], pModes[j].size(),
						 extents[j].data(), NULL, cfloat, CUTENSOR_OP_ABS));

    }
  }
}

// Reduction methods for ASPECTRA integrations (integrating Phi**2, for example, representing an adiabatic ion species)
Red::Red(Grids *grids, std::vector<int> spectra, float dum) : grids_(grids), spectra_(spectra)
{  
  version_red = 'I';
  int J;  J = spectra_.size();
  initialized.assign(J, 0);   desc.resize(J);   sAdd.assign(J, 0);  extents.resize(J);

  extent['y'] = grids_->Nyc;
  extent['x'] = grids_->Nx;
  extent['z'] = grids_->Nz;
  
  for (auto mode : Imode) extent_I.push_back(extent[mode]);
  for (int j = 0; j < J; j++) {
    if (spectra_[j] == 1) {
      for (auto mode : iModes[j]) extents[j].push_back(extent[mode]);

      //      printf("0 =? %d \n",iModes[0].size());
     
      if( !isCuTensorInitialised ) {
         // First time, do initialisation
         HANDLE_ERROR( cutensorInit(&handle));
      }

      HANDLE_ERROR( cutensorInitTensorDescriptor(&handle, &dI, nImode, extent_I.data(), NULL, cfloat, CUTENSOR_OP_ABS));
      HANDLE_ERROR( cutensorInitTensorDescriptor(&handle, &desc[j], iModes[j].size(),
						 extents[j].data(), NULL, cfloat, CUTENSOR_OP_ABS));

    }
  }
}

Red::~Red() {
  if (Addwork) cudaFree(Addwork);
  if (Maxwork) cudaFree(Maxwork);
}


void Red::pSum(float* P2, float* res, int ispec)
{
  // calculate reduction, res = res + sum_i P2(i), over first few indices, mainly for (1-Gamma_0) Phi**2 integrations
  if (version_red != 'P') {printf("version should be P in pSum \n"); exit(1);}

  if (initialized[ispec] == 0) {
  
    HANDLE_ERROR(cutensorReductionGetWorkspace(&handle, P2, &dP, Pmode.data(),
					       res, &desc[ispec], pModes[ispec].data(),
					       res, &desc[ispec], pModes[ispec].data(),
					       opAdd, typeCompute, &sizeAdd));;
    if (sizeAdd > sizeWork) {
      sizeWork = sizeAdd;
      if (Addwork) cudaFree (Addwork);
      if (cudaSuccess != cudaMalloc(&Addwork, sizeWork)) {
	Addwork = nullptr;
	sizeWork = 0;
      }
    }
    initialized[ispec]  = 1;
  }
  
  HANDLE_ERROR(cutensorReduction(&handle,
				 (const void*) &alpha, P2, &dP, Pmode.data(),
				 (const void*) &beta,  res,  &desc[ispec], pModes[ispec].data(),
				                       res,  &desc[ispec], pModes[ispec].data(),
				 opAdd, typeCompute, Addwork, sizeWork, 0));
}
void Red::iSum(float* I2, float* res, int ispec)
{
  if (version_red != 'I') {printf("version should be I in iSum \n"); exit(1);}

  if (initialized[ispec] == 0) {
  
    HANDLE_ERROR(cutensorReductionGetWorkspace(&handle, I2, &dI, Imode.data(),
					       res, &desc[ispec], iModes[ispec].data(),
					       res, &desc[ispec], iModes[ispec].data(),
					       opAdd, typeCompute, &sizeAdd));;
    if (sizeAdd > sizeWork) {
      sizeWork = sizeAdd;
      if (Addwork) cudaFree (Addwork);
      if (cudaSuccess != cudaMalloc(&Addwork, sizeWork)) {
	Addwork = nullptr;
	sizeWork = 0;
      }
    }
    initialized[ispec]  = 1;
  } 

  HANDLE_ERROR(cutensorReduction(&handle,
				 (const void*) &alpha, I2, &dI, Imode.data(),
				 (const void*) &beta,  res,  &desc[ispec], iModes[ispec].data(),
				                       res,  &desc[ispec], iModes[ispec].data(),
				 opAdd, typeCompute, Addwork, sizeWork, 0));
}

void Red::Max(float* A2, float* B)
{
  // calculate reduction, B = max(|A2|), over first few indices
  if (version_red != 'N') {printf("version should be N in Max \n"); exit(1);}

  if (first_Max) {
    
    // get workspace (sizeMax) for a max (opMax) over |P2|
    HANDLE_ERROR(cutensorReductionGetWorkspace(&handle,
					       A2, &dA, Amode.data(),
					       B,  &dB, Bmode.data(),
					       B,  &dB, Bmode.data(),
					       opMax, typeCompute, &sizeMax));
    
    if (cudaSuccess != cudaMalloc(&Maxwork, sizeMax)) {
      Maxwork = nullptr;
      sizeMax = 0;
    }
    first_Max = false;
  }

  HANDLE_ERROR(cutensorReduction(&handle,
				 (const void*) &alpha, A2, &dA, Amode.data(),
				 (const void*) &beta,  B,  &dB, Bmode.data(),
				                       B,  &dB, Bmode.data(),
				 opMax, typeCompute, Maxwork, sizeMax, 0));
}

void Red::sSum(float* Q, float* R)
{
  // calculate reduction, R = sum Q, leaving results sorted by species only
  if (version_red != 'N') {printf("version should be N in sSum \n"); exit(1);}

  if (first_Sum) {
    
    // get workspace (sizeWork) for a sum (opSum) over Q
    HANDLE_ERROR(cutensorReductionGetWorkspace(&handle,
					       Q,  &dQ, Qmode.data(),
					       R,  &dR, Rmode.data(),
					       R,  &dR, Rmode.data(),
					       opAdd, typeCompute, &sizeAdd));
    
    if (cudaSuccess != cudaMalloc(&Addwork, sizeAdd)) {
      Addwork = nullptr;
      sizeAdd = 0;
    }
    first_Sum = false;
  }

  HANDLE_ERROR(cutensorReduction(&handle,
				 (const void*) &alpha, Q, &dQ, Qmode.data(),
				 (const void*) &beta,  R, &dR, Rmode.data(),
				                       R, &dR, Rmode.data(),
				 opAdd, typeCompute, Addwork, sizeAdd, 0));
}

void Red::aSum(float* A, float* B)
{
  // calculate full reduction, B = sum A
  if (version_red != 'N')  {printf("version should be N in aSum \n"); exit(1);}

  if (first_SumT) {
    
    HANDLE_ERROR(cutensorReductionGetWorkspace(&handle,
					       A,  &dA, Amode.data(),
					       B,  &dB, Bmode.data(),
					       B,  &dB, Bmode.data(),
					       opAdd, typeCompute, &sizeAddT));
    
    if (cudaSuccess != cudaMalloc(&AddworkT, sizeAddT)) {
      AddworkT = nullptr;
      sizeAddT = 0;
    }
    first_SumT = false;
  }

  HANDLE_ERROR(cutensorReduction(&handle,
				 (const void*) &alpha, A, &dA, Amode.data(),
				 (const void*) &beta,  B, &dB, Bmode.data(),
				                       B, &dB, Bmode.data(),
				 opAdd, typeCompute, AddworkT, sizeAddT, 0));
}

// res = Sum_appropriately W
void Red::Sum(float* W, float* res, int ispec) 
{
  if (version_red != 'W')  {printf("version should be W in Sum \n"); exit(1);}
  if (initialized[ispec] == 0) {

    // Get size of workspace that will be used, stored in sizeAdd (sizeWork)
    HANDLE_ERROR(cutensorReductionGetWorkspace(&handle, W, &dW, Wmode.data(),
					       res, &desc[ispec], Modes[ispec].data(),
					       res, &desc[ispec], Modes[ispec].data(),
					       opAdd, typeCompute, &sizeAdd));
    
    // if the size is larger than currently allocated (starting with unallocated) space, free
    // the old one (if it is allocated) and allocate the larger space
    // Assume it is fine to use the larger work space freely. 
    //    printf("Workspace allocation: %d \t with size %d \n",ispec,sAdd[ispec]);
    if (sizeAdd > sizeWork) {
      sizeWork = sizeAdd;
      if (Addwork) cudaFree (Addwork);
      if (cudaSuccess != cudaMalloc(&Addwork, sizeWork)) {
	Addwork = nullptr;
	sizeWork = 0;
      }
      //      printf("work size = %d \n", sizeWork);
    }
    initialized[ispec] = 1;
  } 

  //  printf("Reduction: %d \n",ispec);
  HANDLE_ERROR(cutensorReduction(&handle,
				 (const void*) &alpha, W,   &dW,          Wmode.data(),
				 (const void*) &beta,  res, &desc[ispec], Modes[ispec].data(),
				 res,  &desc[ispec], Modes[ispec].data(),
				 opAdd, typeCompute, Addwork, sizeWork, 0));
  // The final argument in this call is the stream used for the calculation
}

