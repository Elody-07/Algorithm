'''
面试题16：数值的整数次方
题目：实现函数double Power(double base, int exponent)，求base的exponent次方。不得使用库函数，同时不需要考虑大数问题。
'''
InvalidInput = False
def Power(base, exponent):
    global InvalidInput
    if base == 0. and exponent < 0:
        InvalidInput = True
        return 0.
    else:
        InvalidInput = False

    if exponent < 0:
        absexponent = -exponent
    else:
        absexponent = exponent
    
    result = PowerWithUnsignedExponent(base, absexponent)
    if exponent < 0:
        return 1. / result 
    else:
        return result

# 常规解法
# def PowerWithUnsignedExponent(base, exponent):
#     res = 1
#     for i in range(1, exponent + 1):
#         res *= base
#     return res

# 高效解法 
def PowerWithUnsignedExponent(base, exponent):
    if exponent == 0:
        return 1
    if exponent == 1:
        return base 
    
    res = PowerWithUnsignedExponent(base, exponent >> 1)
    res *= res 
    if (exponent & 0b1):
        res *= base
    return res

# print(Power(4, 2), InvalidInput)
# print(Power(4, 0), InvalidInput)
# print(Power(4, -2), InvalidInput)

# print(Power(-4, 2), InvalidInput)
# print(Power(-4, 0), InvalidInput)
# print(Power(-4, -2), InvalidInput)

# print(Power(0, -2), InvalidInput)
# print(Power(0, 0), InvalidInput)
# print(Power(0, 2), InvalidInput)




'''
面试题17：打印从1到最大的n位数
题目：输入数字n，按顺序打印出从1到最大的n位十进制数，比如输入3，则打印出1，2，3一直到999。