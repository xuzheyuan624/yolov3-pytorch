#include<TH/TH.h>

int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2, THFloatTensor *output){
    if (!THFloatTensor_isSameSizeAs(input1, input2)){
        long n1 = THFloatTensor_size(input1, 0);
        long n2 = THFloatTensor_size(input2, 0);
        if (n1 < n2){
            THFloatTensor_resizeAs(input1, input2);
        }
        else{
            THFloatTensor_resizeAs(input2, input1);
        }
    }
    THFloatTensor_resizeAs(output, input1);
    THFloatTensor_cadd(output, input1, 1.0, input2);
    return 1;
}

int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input){
    THFloatTensor_resizeAs(grad_input, grad_output);
    THFloatTensor_fill(grad_input, 1);
    return 1;
}

    if (center == 1){
        b1_x1 = bbox1_flat[0] - bbox1_flat[2] / 2;
        b1_x2 = bbox1_flat[0] + bbox1_flat[2] / 2;
        b1_y1 = bbox1_flat[1] - bbox1_flat[3] / 2;
        b1_y2 = bbox1_flat[1] + bbox1_flat[3] / 2;
    }
    else{
        b1_x1 = bbox1_flat[0];
        b1_y1 = bbox1_flat[1];
        b1_x2 = bbox1_flat[2];
        b1_y2 = bbox1_flat[3];
    }
    for (i=0;i<boxes_num;++i){
        if (center == 1){
            b2_x1 = bbox2_flat[i * boxes_dim] - bbox2_flat[i * boxes_dim + 2] / 2;
            b2_x2 = bbox2_flat[i * boxes_dim] + bbox2_flat[i * boxes_dim + 2] / 2;
            b2_y1 = bbox2_flat[i * boxes_dim + 1] - bbox2_flat[i * boxes_dim + 3] / 2;
            b2_y2 = bbox2_flat[i * boxes_dim + 1] + bbox2_flat[i * boxes_dim + 3] / 2;
        }
        else{
            b2_x1 = bbox1_flat[i * boxes_dim];
            b2_y1 = bbox1_flat[i * boxes_dim + 1];
            b2_x2 = bbox1_flat[i * boxes_dim + 2];
            b2_y2 = bbox1_flat[i * boxes_dim + 3];
        }
        inter_rect_x1 = fmaxf(b1_x1, b2_x2);
        inter_rect_x2 = fminf(b1_x2, b2_x2);
        inter_rect_y1 = fmaxf(b1_y1, b2_y1);
        inter_rect_y2 = fminf(b1_y2, b2_y2);

        inter_area = fmaxf(inter_rect_x2 - inter_rect_x1 + 1, 0) * fmaxf(inter_rect_y2 - inter_rect_y1 + 1, 0);

        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1);
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1);

        output_flat[i] = inter_area / (b1_area + b2_area + 1e-16);
    }