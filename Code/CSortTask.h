/******************************************************************************
						 .88888.   888888ba  dP     dP
						 d8'   `88  88    `8b 88     88
						 88        a88aaaa8P' 88     88
						 88   YP88  88        88     88
						 Y8.   .88  88        Y8.   .8P
						 `88888'   dP        `Y88888P'

						 a88888b.                                         dP   oo
						 d8'   `88                                         88
						 88        .d8888b. 88d8b.d8b. 88d888b. dP    dP d8888P dP 88d888b. .d8888b.
						 88        88'  `88 88'`88'`88 88'  `88 88    88   88   88 88'  `88 88'  `88
						 Y8.   .88 88.  .88 88  88  88 88.  .88 88.  .88   88   88 88    88 88.  .88
						 Y88888P' `88888P' dP  dP  dP 88Y888P' `88888P'   dP   dP dP    dP `8888P88
						 88                                        .88
						 dP                                    d8888P
						 ******************************************************************************/

#ifndef _CSORT_TASK_H
#define _CSORT_TASK_H

#include <iostream>
#include <sstream>
#include <algorithm>
#include <math.h>
#include "../Common/IComputeTask.h"

template <typename T>
class CSortTask : public IComputeTask
{
public:
    CSortTask(const size_t ArraySize, size_t LocWorkSize[3]) : m_N(ArraySize), LocalWorkSize() {
        m_N_padded = getPaddedSize(m_N);
        LocalWorkSize[0] = LocWorkSize[0];
        LocalWorkSize[1] = LocWorkSize[1];
        LocalWorkSize[2] = LocWorkSize[2];
    }

	virtual ~CSortTask();

	// IComputeTask
    virtual bool InitResources(cl_device_id Device, cl_context Context) override;
    virtual void ReleaseResources() override;
    virtual void ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]) override;
    virtual void ComputeCPU() override;

	virtual bool ValidateResults();

    std::string toString(const T* p, const unsigned n) const {
        std::stringstream ss;
        for(unsigned i = 0; i < n; ++i)
            ss << p[i] << " ";
        ss << std::endl;
        return ss.str();
    }

protected:
    size_t getPaddedSize(size_t n) const {
        unsigned int log2val = (unsigned int)ceil(log((float)n) / log(2.f));
        return (size_t)pow(2, log2val);
    }

    void StdSort();
    void StdStableSort();
    void MergeSort();
    void TimSort();
    bool ValidateSorted(const T* v); // check sorted till N

	void Sort_Mergesort(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);
	void Sort_SimpleSortingNetwork(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);
	void Sort_SimpleSortingNetworkLocal(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);
	void Sort_BitonicMergesort(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3]);

	void ExecuteTask(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int task);
	void TestPerformance(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int task);

	//NOTE: we have two memory address spaces, so we mark pointers with a prefix
	//to avoid confusions: 'h' - host, 'd' - device

	size_t				m_N;
	size_t				m_N_padded;
	size_t				LocalWorkSize[3];

	// input data
    T*                  m_hInput = nullptr;
	// results
    T*                  m_resultCPU = nullptr;
    T*                  m_resultGPU[3];

    cl_mem				m_dPingArray = nullptr;
    cl_mem				m_dPongArray = nullptr;

	//OpenCL program and kernels
    cl_program			m_Program = nullptr;
    cl_kernel			m_MergesortStartKernel = nullptr;
    cl_kernel			m_MergesortGlobalSmallKernel = nullptr;
    cl_kernel			m_MergesortGlobalBigKernel = nullptr;
    cl_kernel			m_SimpleSortingNetworkKernel = nullptr;
    cl_kernel			m_SimpleSortingNetworkLocalKernel = nullptr;
    cl_kernel			m_BitonicGlobalKernel = nullptr;
    cl_kernel			m_BitonicLocalKernel = nullptr;
    cl_kernel			m_BitonicStartKernel = nullptr;
};

#endif // _CSORT_TASK_H
