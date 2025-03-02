


#include "forward.h"
#include "auxiliary.h"

// 声明设备函数 floatToStr
__device__ void floatToStr(float value, char* buffer, int precision = 4);

// 声明设备函数 custom_strcat
__device__ void custom_strcat(char* dest, const char* src);

// 声明模板函数 glmMatToString
// 注意：模板函数的定义需要在头文件中包含，因此这里使用extern template声明
template <typename T>
__device__ char* glmMatToString(const T& mat, int h, int w);

