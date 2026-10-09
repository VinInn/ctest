//
//  /opt/nvidia/nsight-compute/2025.2.1/ncu --metrics sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,,sm__sass_thread_inst_executed_op_hfma_pred_on.sum ./a.out

#include <cuda_fp16.h>
#include <cstdio>
#include<cstdint>
#include<bit>


__global__ 
void  bar(int16_t x,int16_t y){
  uint32_t a = std::bit_cast<uint16_t>(x);  a = (a<<16) | std::bit_cast<uint16_t>(x);
  uint32_t b = std::bit_cast<uint16_t>(y);  b = (b<<16) | std::bit_cast<uint16_t>(y);
  auto c = __vabsdiffs2(a,b);
  uint16_t w = (c&0xffff0000)>>16;
  uint16_t z = c&0xffff;
  printf("%d,%d,%d,%d\n",x,y,w,z);
  c = __vadd2(a,b);
  w = (c&0xffff0000)>>16;
  z = c&0xffff;
  printf("%d,%d,%d,%d\n",x,y,std::bit_cast<int16_t>(w),std::bit_cast<int16_t>(z));
}

int main() {
 
  bar<<<1,1,0,0>>>(int16_t(1024),int16_t(2048));
  bar<<<1,1,0,0>>>(int16_t(-1024),int16_t(2048));
  bar<<<1,1,0,0>>>(int16_t(-(2048*10)),int16_t(2048));
  bar<<<1,1,0,0>>>(int16_t(-(2048*14)),int16_t(4*2048));
  bar<<<1,1,0,0>>>(int16_t(-(2048*14)),int16_t(6*2048));
 cudaDeviceSynchronize();

}
