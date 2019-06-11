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
'''
# 两种方法：常规和递归，共用PrintNumber
def PrintNumsContinuously_1(n):
    if n <= 0:
        return

    num = [0] * n
    count = 0
    while(not Increment(num)):
        PrintNumber(num)
        count += 1
    print(count)

# num+1, 并判断是否溢出（即最高位由9->10）
def Increment(num):
    length = len(num)
    nTakeOver = 0
    for i in range(length-1, -1, -1):
        nSum = num[i] + nTakeOver
        if i == length - 1:
            nSum += 1
        if nSum >= 10:
            if i == 0:
                return True
            else:
                nTakeOver = 1
                nSum -= 10
        else:
            nTakeOver = 0
        num[i] = nSum
    return False

# 打印num数组代表的数字，不打印高位无意义的0
def PrintNumber(num):
    isBeginning0 = True
    for i in range(0, len(num)):
        if isBeginning0 and num[i] != 0:
            isBeginning0 = False
        
        if not isBeginning0:
            print('%d' % num[i], end='')
    
    if not isBeginning0: # 防止递归方法中多打印一行空行
        print('')


# 递归
def PrintNumsContinuously_2(n):
    if n <= 0:
        return 
    num = [0] * n
    for i in range(0, 10):
        num[0] = i
        PrintNumsCore(num, 0)

def PrintNumsCore(num, index):
    if index == len(num)-1:
        PrintNumber(num)
        return

    for i in range(10):
        num[index+1] = i
        PrintNumsCore(num, index+1)

# PrintNumsContinuously_2(0)
# PrintNumsContinuously_2(-1)
PrintNumsContinuously_2(2)


