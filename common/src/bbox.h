typedef int bool;
#define True 1
#define False 0

int bboxiou(THFloatTensor *bbox1, THFloatTensor *bbox2, THFloatTensor *output, bool center);