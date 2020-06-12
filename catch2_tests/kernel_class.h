
class Kernels {
 public:
  Kernels();
  void kInit(int, int, int, int, int, float*, float*, float*, float, float, float);
  
  float * ky;
  float * kx;
  float * kz;

  float * ky_h;
  float * kx_h;
  float * kz_h;

};

/*
class Red {
 public:
  Red(Grids* grids);
  ~Red();
  void Sum(float* rmom, float *val);
 private:
  float* val;
  float* dum;
  void* work_sum;
  size_t nwork_sum;
  const Grids* grids_;
  const size_t Ng_;
};
*/
