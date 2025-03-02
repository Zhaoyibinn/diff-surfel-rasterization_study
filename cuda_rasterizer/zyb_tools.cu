

#include "forward.h"
#include "auxiliary.h"
__device__ void floatToStr(float value, char* buffer, int precision = 4) {
    // 假设buffer足够大，至少20个字符
    int i = 0;

    // 处理负号
    if (value < 0) {
        buffer[i++] = '-';
        value = -value;
    }

    // 获取整数部分
    int integerPart = (int)value;
    float decimalPart = value - integerPart;

    // 将整数部分转换为字符串
    if (integerPart == 0) {
        buffer[i++] = '0';  // 处理整数部分为0的情况
    } else {
        int temp = integerPart;
        int digits = 0;
        while (temp > 0) {
            temp /= 10;
            digits++;
        }
        temp = integerPart;
        for (int j = digits - 1; j >= 0; j--) {
            buffer[i++] = (temp % 10) + '0';
            temp /= 10;
        }
    }

    // 添加小数点
    buffer[i++] = '.';

    // 将小数部分转换为字符串
    for (int j = 0; j < precision; j++) {
        decimalPart *= 10;
        int digit = (int)decimalPart;
        buffer[i++] = digit + '0';
        decimalPart -= digit;
    }

    // 添加字符串结束符
    buffer[i] = '\0';
}

__device__ void custom_strcat(char* dest, const char* src) {
    while (*dest != '\0') {
        dest++;
    }
    while (*src != '\0') {
        *dest = *src;
        dest++;
        src++;
    }
    *dest = '\0';
}


template <typename T>
__device__ char* glmMatToString(const T& mat,int h, int w) {
    char buffer[1024 * sizeof(char)];  // 假设缓冲区足够大
    buffer[0] = '\0';  // 初始化为空字符串
	// custom_strcat(buffer, "test\n"); 
    int rows = h;
    int cols = w;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // char temp[64];  // 临时缓冲区
            // 将矩阵元素格式化为字符串
            // sprintf(temp, "%.2f\t", mat[i][j]);
			char d_buffer[16 * sizeof(char)];
			floatToStr(mat[i][j], d_buffer);
            custom_strcat(buffer, d_buffer);
			custom_strcat(buffer, "\t");  // 将临时字符串追加到结果中
        }
        custom_strcat(buffer, "\n");  // 每行结束后追加换行符
    }
	custom_strcat(buffer, "\n");
    return buffer;
}