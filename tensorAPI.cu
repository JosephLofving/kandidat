
#include "tensorAPI.h"


void setElement(cuDoubleComplex* tensor, int row, int column, int slice, int matLength, cuDoubleComplex value){
    tensor[row + column*matLength+slice*matLength*matLength] = value;
}

cuDoubleComplex getElement(cuDoubleComplex* tensor, int row, int column, int slice, int matLength){
    return tensor[row + column*matLength+slice*matLength*matLength];
}