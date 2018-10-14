#include<TH/TH.h>
#include<math.h>
#include"bbox.h"

int bboxiou(THFloatTensor *bbox1, THFloatTensor *bbox2, THFloatTensor *output, bool center){
    //THArgCheck(THLongTensor_isContiguous(bbox1), 2, "boxes must be contiguous");
    //THArgCheck(THLongTensor_isContiguous(bbox2), 2, "boxes must be contiguous");
    //number of boxes
    long boxes_num = THFloatTensor_size(bbox2, 0);
    long boxes_dim = THFloatTensor_size(bbox2, 1);

    float *output_flat = THFloatTensor_data(output);

    float b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2;
    float inter_rect_x1, inter_rect_x2, inter_rect_y1, inter_rect_y2;
    float inter_area, b1_area, b2_area;
    float *bbox1_flat = THFloatTensor_data(bbox1);
    float *bbox2_flat = THFloatTensor_data(bbox2);
    int i;

    if (center){
        b1_x1 = bbox1_flat[0] - bbox1_flat[2] / 2;
        b1_x2 = bbox1_flat[0] + bbox1_flat[2] / 2;
        b1_y1 = bbox1_flat[1] - bbox1_flat[3] / 2;
        b1_y2 = bbox1_flat[1] + bbox1_flat[3] / 2;
        //printf('%f\n', b1_x1);
    }
    else{
        b1_x1 = bbox1_flat[0];
        b1_y1 = bbox1_flat[1];
        b1_x2 = bbox1_flat[2];
        b1_y2 = bbox1_flat[3];
    }
    for (i=0;i<boxes_num;++i){
        if (center){
            b2_x1 = bbox2_flat[i * boxes_dim] - bbox2_flat[i * boxes_dim + 2] / 2;
            b2_x2 = bbox2_flat[i * boxes_dim] + bbox2_flat[i * boxes_dim + 2] / 2;
            b2_y1 = bbox2_flat[i * boxes_dim + 1] - bbox2_flat[i * boxes_dim + 3] / 2;
            b2_y2 = bbox2_flat[i * boxes_dim + 1] + bbox2_flat[i * boxes_dim + 3] / 2;
        }
        else{
            b2_x1 = bbox2_flat[i * boxes_dim];
            b2_y1 = bbox2_flat[i * boxes_dim + 1];
            b2_x2 = bbox2_flat[i * boxes_dim + 2];
            b2_y2 = bbox2_flat[i * boxes_dim + 3];
        }
        inter_rect_x1 = fmaxf(b1_x1, b2_x1);
        inter_rect_x2 = fminf(b1_x2, b2_x2);
        inter_rect_y1 = fmaxf(b1_y1, b2_y1);
        inter_rect_y2 = fminf(b1_y2, b2_y2);

        inter_area = fmaxf(inter_rect_x2 - inter_rect_x1 + 1, 0) * fmaxf(inter_rect_y2 - inter_rect_y1 + 1, 0);

        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1);
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1);

        output_flat[i] = inter_area / (b1_area + b2_area - inter_area + 1e-16);
    }

    return 1;
}

