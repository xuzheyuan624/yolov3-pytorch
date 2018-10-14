#include<TH/TH.h>
#include<math.h>

int cpu_nms(THLongTensor *keep_out, THLongTensor *nms_out, THFloatTensor *boxes, THLongTensor *order, THFloatTensor *areas, float conf_thresh, float nms_thresh){
    // boxes must to be sorted
    THArgCheck(THLongTensor_isContiguous(keep_out), 0, "keep_out must be contiguous");
    THArgCheck(THLongTensor_isContiguous(boxes), 2, "boxes must be contiguous");
    THArgCheck(THLongTensor_isContiguous(order), 3, "order must be contiguous");
    THArgCheck(THLongTensor_isContiguous(areas), 4, "areas must be contiguous");
    //batch size
    long batch_size = 
}