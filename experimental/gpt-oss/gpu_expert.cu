// Reference kernels: one warp per output row. No CUDA toolkit headers required.
extern "C" __device__ float weight(const unsigned char* data,int row,int col,int cols,int format) {
    if(format==0) return reinterpret_cast<const float*>(data)[row*cols+col];
    const unsigned char* block=data+(row*(cols/32)+col/32)*17;
    const unsigned char packed=block[1+col%16];
    const int code=(col%32<16) ? packed&15 : packed>>4;
    const float values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
    const unsigned exponent=block[0];
    const unsigned bits=exponent<2 ? 0x00200000u<<exponent : (exponent-1u)<<23;
    return values[code]*__uint_as_float(bits);
}
extern "C" __global__ void gate_up(const unsigned char* gate,const unsigned char* up,
    const float* gb,const float* ub,const float* input,float* hidden,int h,int f,int gf,int uf) {
    const int row=blockIdx.x,lane=threadIdx.x;
    if(row>=f) return;
    float g=0,u=0;
    for(int col=lane;col<h;col+=32) {
        g+=weight(gate,row,col,h,gf)*input[col];
        u+=weight(up,row,col,h,uf)*input[col];
    }
    for(int offset=16;offset;offset/=2) {
        g+=__shfl_down_sync(0xffffffffu,g,offset);
        u+=__shfl_down_sync(0xffffffffu,u,offset);
    }
    if(lane==0) {
        g=fminf(g+gb[row],7.f); u=fminf(fmaxf(u+ub[row],-7.f),7.f);
        hidden[row]=(g/(1.f+expf(-1.702f*g)))*(u+1.f);
    }
}
extern "C" __global__ void down(const unsigned char* weights,const float* bias,
    const float* hidden,float* output,int h,int f,int format) {
    const int row=blockIdx.x,lane=threadIdx.x;
    if(row>=h) return;
    float value=0;
    for(int col=lane;col<f;col+=32) value+=weight(weights,row,col,f,format)*hidden[col];
    for(int offset=16;offset;offset/=2) value+=__shfl_down_sync(0xffffffffu,value,offset);
    if(lane==0) output[row]=value+bias[row];
}
