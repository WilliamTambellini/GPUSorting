/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CSortingMain.h"

#include <iostream>
#include <fstream>

#ifdef NDEBUG
  std::ofstream nullStream("");
  std::ostream& debug = nullStream;
#else
  std::ostream& debug = std::cout;
#endif

using namespace std;

int main(int argc, char** argv)
{
    std::cout << "main:" << std::endl;
	CSortingMain mySortingMain;
	auto success = mySortingMain.EnterMainLoop(argc, argv);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
