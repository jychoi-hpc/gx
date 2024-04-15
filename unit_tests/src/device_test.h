#pragma once
extern __device__ float val_d;

__global__ void get_val(float* val);

void EXPECT_FLOAT_EQ_D(float *a_d, float a_correct, float err=0.);
void EXPECT_FLOAT_EQ_D(float *check_d, float *correct_d, float err=0.);
void EXPECT_FLOAT_EQ_REL_D(float *a_d, float a_correct, float err=0.);
