
class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class LinkNode(object):
    def __init__(self, val):
        self.val = val
        self.next = None



# NO.3 check
def getReplicate(arr):
    if arr == [] or num < 0:
        return -1 
    
    for i in range(len(arr)):
        curNum = arr[i]
        while curNum != i:
            if arr[curNum] == curNum:
                return curNum
            else:
                temp = arr[curNum]
                arr[curNum] = curNum
                arr[i] = temp
    
    return -1


def getReplication(arr):
    if arr == []:
        return -1
    
    n = len(arr)
    start = 1
    end = n - 1
    while (start <= end):
        mid = (start + end) // 2
        count = 0
        for i in range(n):
            if arr[i] >= start and arr[i] <= mid:
                count += 1
        if start == end and count > 1:
            return start
        if count > (mid - start + 1):
            end = mid 
        else:
            start = mid + 1
    
    return -1
    

# print(getReplication([1,2,3,7]))
# print(getReplication([1,2,2,3,3,3,3]))
# print(getReplication([1,2,3,2]))

# NO.4 check
def find2D(arr, rows, cols, num):
    if arr == [] or rows <= 0 or cols <= 0:
        return False
    
    row = 0
    col = cols - 1
    while(row < rows and col >= 0):
        if num == arr[row][col]:
            return True
        elif num > arr[row][col]:
            row += 1
        else:
            col -= 1
    return False

# arr = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
# print(find2D(arr, 4, 3, 1))
# print(find2D(arr, 4, 3, 3))
# print(find2D(arr, 4, 3, 10))
# print(find2D(arr, 4, 3, 12))
# print(find2D(arr, 4, 3, 16))


# NO.5 check
def replaceBlank(string):
    if string == '':
        return string
    
    if isinstance(string, str):
        string = list(string)
    
    count = 0
    length = len(string)
    for i in range(length):
        if string[i] == ' ':
            count += 1
    
    if count == 0:
        return ''.join(string)
    else:
        string.extend([None] * 2 * count)
    
    new = len(string) - 1
    old = length - 1
    while(new >= 0 and old >= 0):
        if string[old] == ' ':
            string[new] = '0'
            string[new-1] = '2'
            string[new-2] = '%'
            new -= 3
        else:
            string[new] = string[old]
            new -= 1
        old -= 1
    return ''.join(string)

# print(replaceBlank('We are happy.'))
# print(replaceBlank('Wearehappy.'))



# NO.6 check
def printLink(pHead):
    if pHead is None:
        return []
    
    res = []
    pNode = pHead
    while pNode is not None:
        res.append(pNode.val)
        pNode = pNode.next
    return res[::-1]

# NO.7 check
def reconstruct(pre, mid):
    if len(pre) != len(mid) or pre == [] or mid == []:
        return None
    
    val = pre[0]
    root = TreeNode(val)
    index = findNum(mid, val)
    if index < 0:
        return None

    left_length = index
    right_length = len(pre) - left_length - 1
    if left_length > 0:
        root.left = reconstruct(pre[1:1+left_length], mid[:left_length])
    if right_length > 0:
        root.right = reconstruct(pre[1+left_length:], mid[index+1:])
    
    return root

def findNum(arr, num):
    for i in range(len(arr)):
        if arr[i] == num:
            return i
    return -1

# head = reconstruct([1,2,4,7,3,5,6,8], [4,7,2,1,5,3,8,6])
# print(head.val)


# NO.8 check
def GetNext(pNode):
    if pNode is None:
        return None

    pNext = None
    if pNode.right is not None:
        pCur = pNode.right
        while pCur.left is not None:
            pCur = pCur.left
        pNext = pCur
    else:
        pCur = pNode
        pParent = pCur.parent
        while (pCur is not None and pParent is not None and pCur is pParent.right):
            pCur = pParent
            pParent = pParent.parent
        pNext = pParent
    return pNext


# NO.9 check
class Queue(object):
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, val):
        self.stack1.append(val)
    
    def pop(self):
        if self.stack1 == [] and self.stack2 == []:
            return None
        elif (self.stack2 == [] and self.stack1 != []):
            while self.stack1 != []:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()


# NO.10 check
def Fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    
    first = 0
    second = 1
    for i in range(1, n):
        fn = first + second
        first = second
        second = fn
    return fn


# NO.11 check
def minNumberInRotateArray(arr):
    if arr == []:
        return None
    
    index1 = 0
    index2 = len(arr) - 1
    while (arr[index1] >= arr[index2]):
        indexMid = (index1 + index2) // 2
        if index2 - index1 == 1:
            return arr[index2]

        if arr[index1] == arr[indexMid] and arr[index2] == arr[indexMid]:
            return Order(arr, index1, index2)

        if arr[indexMid] >= arr[index2]:
            index1 = indexMid
        elif arr[indexMid] <= arr[index2]:
            index2 = indexMid

    return arr[index1]

def Order(arr, start, end):
    small = arr[start]
    for i in range(start, end+1):
        if arr[i] < small:
            small = arr[i]
    return small

# print(minNumberInRotateArray([3,4,5,1,2]))
# print(minNumberInRotateArray([1,2,3,4,5]))
# print(minNumberInRotateArray([1,1,1,0,1]))
# print(minNumberInRotateArray([1,0,1,1,1]))


# NO.12 check
def hasPath(arr, rows, cols, path):
    if arr == [] or path == '' or rows <= 0 or cols <= 0:
        return False
    
    visited = [0] * (rows * cols)
    for row in range(rows):
        for col in range(cols):
            if hasPath_core(arr, row, col, rows, cols, path, visited):
                return True
    
    return False
            

def hasPath_core(arr, row, col, rows, cols, path, visited):
    if path == '':
        return True
    
    find = False
    if row >= 0 and row < rows and \
        col >= 0 and col < cols and \
        visited[row * cols + col] == 0 and \
        arr[row * cols + col] == path[0]:

        visited[row * cols + col] = 1
        find = hasPath_core(arr, row-1, col, rows, cols, path[1:], visited) or \
             hasPath_core(arr, row+1, col, rows, cols, path[1:], visited) or \
             hasPath_core(arr, row, col-1, rows, cols, path[1:], visited) or \
             hasPath_core(arr, row, col+1, rows, cols, path[1:], visited)

        if not find:
            visited[row * cols + col] = 0

    return find

# import numpy as np
# arr = list('ABCEHJIGSFCSLOPQADEEMNOEADIDEJFMVCEIFGGS')
# arr_np = np.array(arr).reshape(5,8)
# print(arr_np)
# print(hasPath(arr, 5,8, 'SGGFIECVAASABCEHJIGQEM'))


# NO.13 check
def MovingCount(rows, cols, k):
    if rows <= 0 or cols <= 0 or k < 0:
        return 0
    
    visited = [0] * rows * cols
    count = MovingCount_core(rows, cols, 0, 0, k, visited)

    return count

def MovingCount_core(rows, cols, row, col, k, visited):
    count = 0
    if row >= 0 and row < rows and \
       col >= 0 and col < cols and \
       SumOfNum(row) + SumOfNum(col) <= k and \
       visited[row*cols + col] == 0:

       visited[row*cols+col] = 1

       count = 1 + MovingCount_core(rows, cols, row+1, col, k, visited) + MovingCount_core(rows, cols, row-1, col, k, visited) + MovingCount_core(rows, cols, row, col-1, k, visited) + MovingCount_core(rows, cols, row, col+1, k, visited)


    return count


def SumOfNum(num):
    sum = 0
    while(num >= 1):
        sum += int(num % 10.)
        num /= 10.
    return sum

# print(MovingCount(1,100,10))


# NO.14 check
def cutRope(num):
    if num <= 0 :
        return 0
    elif num == 1:
        return 1
    elif num == 2:
        return 1
    elif num == 3:
        return 2
    
    product = [0] * (num+1)
    product[1] = 1
    product[2] = 2
    product[3] = 3
    for i in range(4, num+1):
        max = 0
        for j in range(1, int(i//2)+1):
            if product[j] * product[i-j] > max:
                max = product[j] * product[i-j]
        product[i] = max
    return product[num]


def cutRope(num):
    if num <= 0 :
        return 0
    elif num == 1:
        return 1
    elif num == 2:
        return 1
    elif num == 3:
        return 2
    
    if num % 3 == 1:
        timeOf2 = 2
        timeOf3 = int((num - 4) / 3)
    elif num % 3 == 2:
        timeOf2 = 1
        timeOf3 = int((num - 2) / 3)
    else:
        timeOf2 = 0
        timeOf3 = int(num / 3)
    return 2**timeOf2 * 3 ** timeOf3

# print(cutRope(5))
    

# NO.15 check 牛客网的答案是补码包含的1的个数
def NumberOf1(num):
    if num < 0:
        num = num & 0xffffffff
    count = 0
    while(num != 0):
        num = (num-1) & num
        count += 1
    return count


# check
# 设num是32位
def NumberOf1(num):
    return sum([(num >> i & 1) for i in range(0, 32)] )


# NO.16 check
def Power(base, exponent):
    if base == 0 and exponent < 0:
        return False
    
    flag = 1
    if exponent < 0:
        flag = -1
    
    result = PowerUnsigned(base, flag * exponent)
    if exponent < 0:
        return 1. / result
    else:
        return result

def PowerUnsigned(base, exponent):
    if exponent == 0:
        return 1
    
    res = PowerUnsigned(base, exponent >> 1)
    if exponent & 0b1 :
        res = res * res * base
    else:
        res *= res
    return res
    

# NO.17
def printMaxN(n):
    if n <= 0:
        return None

    arr = [0] * n
    printMaxN_core(arr, 0)

def printMaxN_core(arr, idx):
    if idx >= len(arr):
        printNumber(arr)
        return
    
    for i in range(10):
        arr[idx] = i
        printMaxN_core(arr, idx + 1)

def printNumber(arr):
    if arr == []:
        return None
    
    for i in range(len(arr)):
        if arr[i] != 0:
            break

    if i == len(arr) - 1 and arr[i] == 0:
        return 
    else:
        print(''.join(str(x) for x in arr[i:]))

# printMaxN(2)

# NO.18
def DeleteNode(pHead, pNode):
    if pHead is None or pNode is None:
        return None

    if pNode.next is None:
        if pHead is pNode:
            return None

        pTemp = pHead
        while pTemp.next is not None and pTemp.next is not pNode:
            pTemp = pTemp.next
        if pTemp.next is pNode:
            pTemp.next = None
    else:
        pTemp = pNode.next
        pNode.val = pTemp.val
        pNode.next = pTemp.next
        pTemp.next = None
    return pHead
        
# 递归法 check
def DeleteDuplication(pHead):
    if pHead is None or pHead.next is None:
        return pHead
    
    pNext = pHead.next
    if pHead.val != pNext.val :
        pHead.next = DeleteDuplication(pNext)
    else:
        while(pNext is not None and pHead.val == pNext.val):
            pNext = pNext.next
        pHead = DeleteDuplication(pNext)
    
    return pHead
    
    
# NO.19 check
def match(string, pattern):
    if string == '' and pattern == '':
        return True
    if string != '' and pattern == '':
        return False
    
    if len(pattern) > 1 and pattern[1] == '*':
        if len(string) > 0 and (string[0] == pattern[0] or pattern[0] == '.'):
            return match(string, pattern[2:]) or match(string[1:], pattern[2:]) or match(string[1:], pattern)
        else:
            return match(string, pattern[2:])
    
    elif len(string) > 0 and (pattern[0] == '.' or string[0] == pattern[0]):
        return match(string[1:], pattern[1:])
    
    return False

# print(match('', 'b*'))
# print(match('', 'b**'))
# print(match('', '.'))

def isNumeric(string):
    if string == '':
        return False
    
    numberic = True

    dot_idx = findStr(string, '.')
    sci_idx = max(findStr(string, 'e'), findStr(string, 'E'))

    if dot > 0:
        integer = string[:, dot_idx]
        if integer[0] == '-' or integer[0] == '+':
            integer = integer[1:]
        numberic &= scanUnsingedInteger(integer)
    if sci_idx > 0 and 
             
def split(string, dot_idx, sci_idx):
    if sci_idx == len(string) - 1:

    integer, dot, sci = '', '', ''
    if dot_idx > 0:
        integer = string[:dot_idx]
        if sci_idx > 0:
            dot = string[dot_idx+1 : sci_idx]
            sci = string[sci_idx+1 :]
    elif dot_idx < 0 and sci_idx > 0:

    if sci_idx > 0 and dot_idx > 0:
        sci = string[sci_idx+1 :dot_idx]
    elif sci_idx > 0 and dot_idx < 0:
        sci = string


def findStr(string, target):
    for i in range(len(string)):
        if string[i] == target:
            break

    if string[i] == target:
        return i
    else:
        return -1

def scanUnsingedInteger(string):
    for s in string:
        if s < '0' or s > '9':
            return False
