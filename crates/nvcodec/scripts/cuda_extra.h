/*
 * CUDA driver entry points ferricast needs that FFmpeg's dynlink_cuda.h
 * does not declare. Signatures follow NVIDIA's cuda.h / cudaEGL.h
 * (driver API reference); only the function types are declared, the
 * symbols are resolved at runtime.
 */
#ifndef FERRICAST_CUDA_EXTRA_H
#define FERRICAST_CUDA_EXTRA_H

/* EGLImageKHR, prefixed like dynlink_cuda.h does for its EGL types. */
typedef void *ffnv_EGLImageKHR;

typedef CUresult CUDAAPI tcuCtxSetCurrent(CUcontext ctx);
typedef CUresult CUDAAPI tcuMemGetInfo_v2(size_t *free, size_t *total);
typedef CUresult CUDAAPI tcuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);

typedef CUresult CUDAAPI tcuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N);
typedef CUresult CUDAAPI tcuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
typedef CUresult CUDAAPI tcuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
typedef CUresult CUDAAPI tcuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);

typedef CUresult CUDAAPI tcuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);

typedef CUresult CUDAAPI tcuGraphicsEGLRegisterImage(CUgraphicsResource *pCudaResource, ffnv_EGLImageKHR image, unsigned int flags);
typedef CUresult CUDAAPI tcuGraphicsResourceGetMappedEglFrame(CUeglFrame *eglFrame, CUgraphicsResource resource, unsigned int index, unsigned int mipLevel);
typedef CUresult CUDAAPI tcuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags);

#endif
