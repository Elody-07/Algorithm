class LinkNode(object):
    def __init__(self, val):
        self.val = val
        self.next = None

'''
面试题39：数组中出现次数超过一半的数字
题目：数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如，输入一个长度为9的数组[1,2,3,2,2,2,5,4,2]。由于数字2在数组中出现了5次，超过数组长度的一半，故输出2。
牛客网check(2种方法)
'''
def MoreThanHalfTimes1(arr):
    n = len(arr)
    if n <= 0:
        return 0
    
    middle = n >> 1
    start = 0
    end = n - 1
    index = Partition(arr, start, end)
    while(index != middle):
        if index < middle:
            index = Partition(arr, index+1, end)
        else:
            index = Partition(arr, start, index-1)
    
    result = arr[index]
    if not checkMoreThanHalf(arr, result):
        result = 0
    
    return result

def Partition(arr, start, end):
    center = arr[start]
    while(start < end):
        while arr[end] >= center and start < end:
            end -= 1
        arr[start] = arr[end]
        while arr[start] < center and start < end:
            start += 1
        arr[end] = arr[start]
    assert start == end
    arr[start] = center
    return start

def MoreThanHalfTimes2(arr):
    n = len(arr)
    if n <= 0:
        return 0
    
    result = arr[0]
    times = 1
    for i in range(1, n):
        if times == 0:
            result = arr[i]
            times = 1
            continue
        if arr[i] != result:
            times -= 1
        else:
            times += 1
    
    if not checkMoreThanHalf(arr, result):
        result = 0
    return result



def checkMoreThanHalf(arr, result):
    times = 0
    for item in arr:
        if item == result:
            times += 1
    
    if times*2 > len(arr):
        return True
    else:
        return False

# print(MoreThanHalfTimes2([1,3,2,5,3,6]))
        
        
        
'''
面试题40：最小的k个数
题目：输入n个整数，找出其中最小的k个数。例如，输入[4,5,1,6,2,7,3,8]，则最小的4个数字是[1,2,3,4]
牛客网check(2种方法)
'''
def GetLeastNumbers1(arr, k):
    if len(arr) < k or arr == [] or k < 1:
        return []
    if len(arr) == k:
        return arr
    
    start = 0
    end = len(arr) - 1
    index = Partition(arr, start, end)
    while(index != k - 1):
        if index < k - 1:
            start = index + 1
            index = Partition(arr, start, end)
        if index > k - 1:
            end = index - 1
            index = Partition(arr, start, end)
    
    return sorted(arr[:index+1])
    
def GetLeastNumbers2(arr, k):
    if len(arr) < k or arr == [] or k < 1:
        return []
    if len(arr) == k:
        return arr
    least = [0] + arr[:k]
    for i in range(k//2, 0, -1):
        Sift(least, i, k)

    for index in range(k, len(arr)):
        if arr[index] < least[1]:
            least[1] = arr[index]
            Sift(least, 1, k)
    
    return least[1:]

def Sift(arr, start, end):
    j = 2 * start
    while(j <= end):
        if j < end and arr[j] < arr[j+1]:
            j += 1
        if arr[start] < arr[j]:
            temp = arr[j]
            arr[j] = arr[start]
            arr[start] = temp
            start = j 
            j = 2 * start
        else:
            break
    return arr


# arr = [4,5,1,6,2,7,3,8]
# print(GetLeastNumbers1(arr, 8))
# print(GetLeastNumbers2(arr, 8))
# print(arr)


'''
面试题41：数据流中的中位数
题目：如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数是所有数值排序后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数是所有数值排序后中间两个数的平均值。
牛客网check
'''
class DynamicArray(object):

    def __init__(self):
        self.data_min = [None]
        self.data_max = [None]
    
    def Insert(self, value):
        # value是数据流中的第奇数个值，插入大根堆
        if (len(self.data_min) + len(self.data_max)) & 1 == 0:
            if len(self.data_min) > 1 and value > self.data_min[1]:
                self.data_min.insert(1, value)
                self.sift_min(self.data_min, 1, len(self.data_min)-1 )

                value = self.data_min[1]
                self.data_min[1] = self.data_min[-1]
                self.data_min.pop()
                self.sift_min(self.data_min, 1, len(self.data_min)-1 )
            
            self.data_max.insert(1, value)
            self.sift_max(self.data_max, 1, len(self.data_max)-1)
        else:
            if len(self.data_max) > 1 and value < self.data_max[1]:
                self.data_max.insert(1, value)
                self.sift_max(self.data_max, 1, len(self.data_max)-1)

                value = self.data_max[1]
                self.data_max[1] = self.data_max[-1]
                self.data_max.pop()
                self.sift_max(self.data_max, 1, len(self.data_max)-1 )
            
            self.data_min.insert(1, value)
            self.sift_min(self.data_min, 1, len(self.data_min)-1)
    
    def GetMedian(self):
        size = len(self.data_min) + len(self.data_max)
        if size <= 2:
            return None

        if size & 1 == 0:
            median = (self.data_min[1] + self.data_max[1])/2.
        else:
            median = self.data_max[1]
        
        return median
    
    def sift_max(self, arr, start, end):
        j = 2 * start
        while(j <= end):
            if j < end and arr[j+1] > arr[j]:
                j += 1
            if arr[start] < arr[j]:
                temp = arr[j]
                arr[j] = arr[start]
                arr[start] = temp
                start = j
                j = 2*start
            else:
                break

    def sift_min(self, arr, start, end):
        j = 2 * start
        while(j <= end):
            if j < end and arr[j+1] < arr[j]:
                j += 1
            if arr[start] > arr[j]:
                temp = arr[j]
                arr[j] = arr[start]
                arr[start] = temp
                start = j
                j = 2*start
            else:
                break

# dataflow = DynamicArray()
# dataflow.Insert(5)
# print(dataflow.GetMedian())
# dataflow.Insert(2)
# print(dataflow.GetMedian())


'''
面试题42：连续子数组的最大和
题目：输入一个整形数组，数组里既有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。要求时间复杂度为O(n)。
牛客网check
'''
def FindGreatestSumOfSubArray(arr):
    if arr == []:
        return 0
    
    curSum = arr[0]
    greatest = curSum
    for i in range(1, len(arr)):
        if curSum <= 0:
            curSum = arr[i]
        else:
            curSum += arr[i]
        
        if curSum > greatest:
            greatest = curSum

    return greatest

# print(FindGreatestSumOfSubArray([1,-2,3,10,-4,7,2,-5]))


'''
面试题43：1~n整数中1出现的次数
题目：输入一个整数n，求1~n这n个整数的十进制表示中1出现的次数。例如，输入12，1~12这些整数中包含1的数字有1、10、11、12，1一共出现了5次。
牛客网check
'''
def NumberOf1(num):
    if isinstance(num, int):
        num = str(num)
    
    if num == '' or num[0] < '0' or num[0] > '9':
        return 0
    
    first = int(num[0]) 
    length = len(num)
    if length == 1 and first == 0:
        return 0
    if length == 1 and first > 0:
        return 1
    
    numFirstDigit = 0 # 最高位是1（总位数是length）
    if first > 1:
        numFirstDigit = 10 ** (length-1)
    elif first == 1:
        numFirstDigit = int(num[1:]) + 1
    
    numOtherDigit = first * (length-1) * (10 ** (length-2))
    numRecursive = NumberOf1(num[1:])

    return numFirstDigit + numOtherDigit + numRecursive

# print(NumberOf1(0))
# print(NumberOf1(1))
# print(NumberOf1(1000))
    

'''
面试题44：数字序列中某一位的数字
题目：数字以0123456789101112131415……的格式化序列到一个字符序列中。在这个序列中，从0开始计数，第5位是5，第13位是1，第19位是4……求任意第n位对应的数字。
'''
def digitAtIndex(index):
    if index < 0:
        return -1
    
    digits = 1
    while True:
        numbers = 10 if digits == 1 else digits * 9 * 10 ** (digits-1)
        if index < numbers:
            return digitAtIndexCore(index, digits)
        index -= numbers
        digits += 1

def digitAtIndexCore(index, digits):
    begin = 0 if digits == 1 else 10 ** (digits-1)
    num = begin + index / digits
    return int(str(num)[index % digits])

# print(digitAtIndex(0))
# print(digitAtIndex(1))
# print(digitAtIndex(5))
# print(digitAtIndex(13))
# print(digitAtIndex(19))



'''
面试题45：把数组排成最小的数
题目：输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如，输入数组{3，32，321}，则打印出这3个数字能排成的最小数字321323。
牛客网check（2种）
'''
def PrintMinNumber1(arr):
    if arr == []:
        return -1
    
    res = []
    begin = 0
    PrintMinNumber1_core(arr, begin, res)
    # print(res)

    return min(res)

def PrintMinNumber1_core(arr, begin, res):
    if begin == len(arr) - 1:
        res.append(list2num(arr))
        return
    
    for i in range(begin, len(arr)):
        temp = arr[begin]
        arr[begin] = arr[i]
        arr[i] = temp
        PrintMinNumber1_core(arr, begin+1, res)
        temp = arr[begin]
        arr[begin] = arr[i]
        arr[i] = temp



from functools import cmp_to_key
def PrintMinNumber2(arr):
    new = sorted(arr, key=cmp_to_key(sorting))
    return list2num(new)

def sorting(num1, num2):
    combine1 = int(str(num1) + str(num2))
    combine2 = int(str(num2) + str(num1))

    if combine1 < combine2:
        return -1
    elif combine1 > combine2:
        return 1
    else:
        return 0

def list2num(arr):
    s = ''
    for item in arr:
        s += str(item)
    return int(s)

# print(PrintMinNumber2([3,32,321]))



'''
面试题46：把数字翻译成字符串
题目：给定一个数字，我们按照如下规则把它翻译成字符串：0->'a', 1->'b', ..., 25->'z'。一个数字可能有多个翻译。例如，12258有5种不同的翻译，分别是'bccfi', 'bwfi', 'bczi', 'mcfi'和'mzi'。编程实现函数，计算一个数字有多少种不同的翻译方法。
'''
def GetTranslationCount(num):
    if num < 0:
        return None
    if isinstance(num, int):
        num = str(num)
    
    counts = [0] * len(num)
    for i in range(len(num)-1, -1, -1):
        if i == len(num) - 1:
            count = 1
        else:
            count = counts[i+1]
        
        if i < len(num) - 1:
            if int(num[i:i+2]) <= 25 and int(num[i:i+2]) >= 10:
                if i < len(num) - 2:
                    count += counts[i+2]
                else:
                    count += 1
        
        counts[i] = count

    return counts[0]
        
    
# print(GetTranslationCount(0))
# print(GetTranslationCount(25))
# print(GetTranslationCount(26))
# print(GetTranslationCount(12258))


'''
面试题47：礼物的最大价值
题目：在一个m*n的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或向下移动一格，直到到达棋盘的右下角。给定一个棋盘及其上面的礼物，请计算你最多能拿到多少价值的礼物？
'''
def getMaxValue1(mat, rows, cols):
    if mat == [] or rows <= 0 or cols <= 0:
        return 0
    
    val = 0
    res = []
    getMaxValue1_core(mat, rows, cols, 0, 0, val, res)

    # print('length: ', len(res))

    return max(res)

def getMaxValue1_core(mat, rows, cols, row, col, val, res):
    if row == rows-1 and col == cols-1:
        val += mat[row][col]
        res.append(val)
        return
    
    if row < rows - 1:
        getMaxValue1_core(mat, rows, cols, row+1, col, val + mat[row][col], res)
    if col < cols - 1:
        getMaxValue1_core(mat, rows, cols, row, col+1, val + mat[row][col], res)


def getMaxValue2(mat, rows, cols):
    if mat == [] or rows <= 0 or cols <= 0:
        return 0
    
    maxValues = [[0] * cols] * rows

    for i in range(rows):
        for j in range(cols):
            left = 0
            up = 0

            if i > 0:
                left = maxValues[i-1][j]
            if j > 0:
                up = maxValues[i][j-1]
            maxValues[i][j] = max(left, up) + mat[i][j]
    
    return maxValues[rows-1][cols-1]


def getMaxValue3(mat, rows, cols):
    if mat == [] or rows <= 0 or cols <= 0:
        return 0
    
    maxValues = [0] * cols
    for i in range(rows):
        for j in range(cols):
            left = 0
            up = 0

            if i > 0:
                up = maxValues[j]
            if j > 0:
                left = maxValues[j-1]
            
            maxValues[j] = max(up, left) + mat[i][j]
    
    return maxValues[cols-1]


# print(getMaxValue1([[1]], 1, 1))
# print(getMaxValue1([[1, 2, 3]], 1, 3))
# print(getMaxValue1([[1],[2],[3]], 3, 1))
# print(getMaxValue1([[1,2],[3,4]], 2, 2))
# print(getMaxValue1([[1,10,3],[12,2,9],[5,7,4],[3,7,16]], 3, 3))
# print(getMaxValue1([[1,10,3,8],[12,2,9,6],[5,7,4,11],[3,7,16,5]], 4, 4))
# print(getMaxValue1([[1,10,3,8,10],[12,2,9,6,10],[5,7,4,11,10],[3,7,16,5,10],[1,2,3,4,5]], 5, 5))


'''
面试题48：最长不含重复字符的子字符串
题目：请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。假设字符串中只包含'a'~'z'的字符。例如，在字符串'arabcacfr'中，最长的不含重复字符的子字符串是'acfr'，长度为4。
'''
def longestSubstring1(string):
    if string == '' or not isinstance(string, str):
        return 0
    
    longest = ''
    max = ''
    for s in string:
        if s not in max:
            max += s
        else:
            max = s
        if len(max) > len(longest):
            longest = max
    
    return len(longest)

def longestSubstring2(string):
    if string == '' or not isinstance(string, str):
        return 0
    
    curLength = 0
    maxLength = 0
    position = [-1] * 26

    for i in range(len(string)):
        prevIndex = position[ord(string[i])-ord('a')]
        if prevIndex < 0 or (i - prevIndex) > curLength:
            curLength += 1
        else:
            if curLength > maxLength:
                maxLength = curLength
            curLength = i - prevIndex
        position[ord(string[i])-ord('a')] = i
    
    if curLength > maxLength:
        maxLength = curLength
        
    return maxLength
    
# print(longestSubstring2('arabcacfr'))
# print(longestSubstring2('a'))
# print(longestSubstring2('aaaa'))
# print(longestSubstring2('abcdef'))


'''
面试题49：丑数
题目：把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
'''
# 耗时过长
def GetUglyNumber1(index):
    if index <= 0:
        return 0
    
    number = 0
    found = 0
    while(found < index):
        number += 1
        if isUgly(number):
            found += 1
    
    return number

def isUgly(number):
    while(number % 2 == 0):
        number /= 2
    while(number % 3 == 0):
        number /= 3
    while(number % 5 == 0):
        number /= 5
    return (number == 1)

def GetUglyNumber2(index):
    if index <= 0:
        return 0
    
    uglyNumbers = [None] * index
    uglyNumbers[0] = 1
    nextUglyIndex = 1

    i2 = 0
    i3 = 0
    i5 = 0

    while nextUglyIndex < index :
        min_num = min(uglyNumbers[i2]*2, uglyNumbers[i3]*3, uglyNumbers[i5]*5)
        uglyNumbers[nextUglyIndex] = min_num

        while (uglyNumbers[i2]*2 <= uglyNumbers[nextUglyIndex]):
            i2 += 1
        while (uglyNumbers[i3]*3 <= uglyNumbers[nextUglyIndex]):
            i3 += 1
        while (uglyNumbers[i5]*5 <= uglyNumbers[nextUglyIndex]):
            i5 += 1
        
        nextUglyIndex += 1
    
    ugly = uglyNumbers[index-1]
    return ugly

# print(GetUglyNumber2(1))
# print(GetUglyNumber2(2))
# print(GetUglyNumber2(11))
    


'''
面试题50：第一个只出现一次的字符
题目一：字符串中第一个只出现一次的字符。
牛客网check
'''
def FirstNotRepeatingChar(string):
    if string == '':
        return -1
    
    hash = [0] * 256
    for s in string:
        hash[ord(s)] += 1
    
    for i in range(len(string)):
        if hash[ord(string[i])] == 1:
            return i
    return -1

# print(FirstNotRepeatingChar('abcd'))
# print(FirstNotRepeatingChar('aaa'))
# print(FirstNotRepeatingChar('aaabbb'))
# print(FirstNotRepeatingChar(''))

'''
题目二：字符流中第一个只出现一次的字符。
例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"；当从该字符流中读出前6个字符"google"时，第一个只出现一次的字符是"l"。
'''
def FirstAppearingOnce(string):
    occurence = [-1] * 256

    for i in range(len(string)):
        if occurence[ord(string[i])] == -1:
            occurence[ord(string[i])] = i 
        else:
            occurence[ord(string[i])] = -2

    minIndex = 256 
    ch = ''
    for i in range(len(occurence)):
        if occurence[i] >=0 and occurence[i] < minIndex:
            minIndex = occurence[i]
            ch = chr(i)
    
    return ch 

# print(FirstAppearingOnce('go'))
# print(FirstAppearingOnce('goo'))
# print(FirstAppearingOnce('goog'))
# print(FirstAppearingOnce('google'))
    

'''
面试题51：数组中的逆序对
题目：在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数。例如，在数组[7,5,6,4]中，一共存在5个逆序对。
牛客网有问题
'''
# 耗时过长
def InversePairs1(arr):
    if arr == []:
        return 0
    
    count = 0
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] > arr[j]:
                count += 1
    return count


from copy import copy
def InversePairs2(arr):
    if arr == []:
        return 0
    
    dup = copy(arr)
    count = InversePairs2_core(arr, dup, 0, len(arr)-1)

    return count

def InversePairs2_core(arr, dup, start, end):
    if start == end:
        # dup[start] = arr[start]
        return 0

    length = (end - start) // 2

    left = InversePairs2_core(dup, arr, start, start + length)
    right = InversePairs2_core(dup, arr, start+length+1, end)

    i = start + length
    j = end
    indexCopy = end
    count = 0
    while(i >= start and j >= start+length+1 and indexCopy >= 0):
        if arr[i] > arr[j]:
            count += j - (start+length+1) + 1
            dup[indexCopy] = arr[i]
            indexCopy -= 1
            i -= 1
        else:
            dup[indexCopy] = arr[j]
            indexCopy -= 1
            j -= 1
    
    while(i >= start and indexCopy >= 0):
        dup[indexCopy] = arr[i]
        indexCopy -= 1
        i -= i

    while(j >= start + length + 1 and indexCopy >= 0):
        dup[indexCopy] = arr[j]
        indexCopy -= 1
        j -= 1
    
    return left + right + count


# print(InversePairs2([7,5,6,4]))
# print(InversePairs2([4,5,6,7]))
# print(InversePairs2([7,6,5,4]))
# print(InversePairs2([7]))
# print(InversePairs2([7,6]))
# print(InversePairs2([6,7]))


'''
面试题52：两个链表的第一个公共节点
题目：输入两个链表，找出它们的第一个公共结点。
牛客网check（2种）
'''
# 时间、空间复杂度均为O(m+n)
def FindFirstCommonNode1(pHead1, pHead2):
    if pHead1 is None or pHead2 is None:
        return None
    stack1 = []
    stack2 = []
    
    pNext = pHead1
    while(pNext is not None):
        stack1.append(pNext)
        pNext = pNext.next
    pNext = pHead2
    while(pNext is not None):
        stack2.append(pNext)
        pNext = pNext.next
    
    common = None
    while(stack1 != [] and stack2 != []):
        top1 = stack1.pop()
        top2 = stack2.pop()
        if top1 == top2:
            common = top1
    return common


# 时间O(m+n)，空间O(1)
def FindFirstCommonNode2(pHead1, pHead2):
    if pHead1 is None or pHead2 is None:
        return None
    
    length1 = GetLinkLength(pHead1)
    length2 = GetLinkLength(pHead2)
    
    lengthDif = abs(length1 - length2)
    if length1 > length2:
        pHeadLong = pHead1
        pHeadShort = pHead2
    else:
        pHeadLong = pHead2
        pHeadShort = pHead1
    
    for i in range(lengthDif):
        pHeadLong = pHeadLong.next
    
    while(pHeadLong != pHeadShort and pHeadLong is not None and pHeadShort is not None):
        pHeadLong = pHeadLong.next
        pHeadShort = pHeadShort.next
    
    if pHeadLong == pHeadShort:
        return pHeadLong
    else:
        return None

def GetLinkLength(pHead):
    if pHead is None:
        return 0
    length = 0
    while (pHead is not None):
        length += 1
        pHead = pHead.next
    return length


