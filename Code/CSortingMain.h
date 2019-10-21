#ifndef _CASSIGNMENT5_H
#define _CASSIGNMENT5_H

#include "../Common/CAssignmentBase.h"

class CSortingMain : public CAssignmentBase
{
public:
	virtual ~CSortingMain() {};

    virtual bool DoCompute(const unsigned n, const std::string type) override;
};

#endif // _CASSIGNMENT5_H
