#include "CSortingMain.h"
#include "CSortTask.h"
#include <iostream>

using namespace std;

bool CSortingMain::DoCompute(const unsigned n, const std::string type) {
	cout << "Running sort task..." << endl << endl;
	{
		// set work size and size of input array
		size_t LocalWorkSize[3] = { 256, 1, 1 };
        unsigned int arraySize = n; // 1024 * 256;

		// info output
		cout << "Start sorting array of size " << arraySize;
		cout << " using LocalWorkSize " << LocalWorkSize[0] << endl << endl;
		// create sorting task and start it
        if (type== "uint") {
            CSortTask<unsigned int> sorting(arraySize, LocalWorkSize);
            RunComputeTask(sorting, LocalWorkSize);
        }
        else if (type=="f32") {
            CSortTask<float> sorting(arraySize, LocalWorkSize);
            RunComputeTask(sorting, LocalWorkSize);
        }
	}

	return true;
}
