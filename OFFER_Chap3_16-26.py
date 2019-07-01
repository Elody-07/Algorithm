class LinkNode(object):
    def __init__(self, value):
        self.value = value
        self.next = None

class TreeNode(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

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
# PrintNumsContinuously_2(2)


'''
面试题18：删除链表的节点
题目一：在O(1)的时间内删除链表节点。
给定单向链表的头指针和一个节点指针，定义一个函数在O(1)时间内删除该节点。链表节点与函数的定义如下：
'''
def DeleteNode(pHead, pToBeDeleted):

    if (pHead is None or pToBeDeleted is None):
        raise RuntimeError("Invalid Input.")

    if pToBeDeleted.next is None: # 被删除的节点是尾节点
        if pHead is pToBeDeleted:
            delete = pToBeDeleted.value
            pHead = None

        pTemp = pHead
        while pTemp.next is not None and pTemp.next is not pToBeDeleted:
            pTemp = pTemp.next
        if pTemp.next is pToBeDeleted:
            pTemp.next = None
        # else:
        #     return False
    else:
        pTemp = pToBeDeleted.next
        pToBeDeleted.value = pTemp.value
        pToBeDeleted.next = pTemp.next
        pTemp.next = None

    return pHead

# a = LinkNode(1)
# b = LinkNode(2)
# c = LinkNode(3)
# d = LinkNode(4)
# a.next, b.next = b, c
# print(DeleteNode(a, a.next))
# print(DeleteNode(a, a.next))
# print(DeleteNode(a, d))
# print(DeleteNode(a, a))
# print(DeleteNode(a, a))
        

'''
题目二：删除链表中重复的节点
在一个排序的链表中，删除重复的节点。如 1 -> 2 -> 3 -> 3 -> 4 -> 4 -> 5删除重复的节点后变成1 -> 2 -> 5
牛客网check
'''
# 递归法，书上P123非递归法
def DeleteDuplication(pHead):
    if pHead is None or pHead.next is None:
        return pHead
    
    pNext = pHead.next
    if pNext.value != pHead.value:
        pHead.next = DeleteDuplication(pNext) # 保留pHead
    else:
        # 退出循环时pNext=None或pNext.value != pHead.value
        while(pNext is not None and pNext.value == pHead.value): 
            pNext = pNext.next
        # pHead.next = DeleteDuplication(pNext)  # 保留一个重复数字
        pHead = DeleteDuplication(pNext)  # 不保留重复数字

    return pHead

# a, b, c, d = LinkNode(1), LinkNode(2), LinkNode(2), LinkNode(2)
# a.next, b.next, c.next = b, c, d
# head = DeleteDuplication(a)
# print(head)
    


'''
面试题19：正则表达式匹配
题目：请实现一个函数用来匹配包含'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，'*'表示它前面的字符可以出现任意次（含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串'aaa'与模式'a.a'和'ab*ac*a'匹配，但与'aa.a'和'ab*a'均不匹配。
牛客网check
'''
def match(target, pattern):
    if len(target) == 0 and len(pattern) == 0:
        return True
    if len(target) > 0 and len(pattern) == 0:
        return False

    if len(pattern) > 1 and pattern[1] == '*':
        if len(target)>0 and (target[0]==pattern[0] or pattern[0]=='.'): 
            return match(target, pattern[2:]) or match(target[1:], pattern[2:]) or match(target[1:], pattern)
        else:
            return match(target, pattern[2:])
    
    if len(target)>0 and (pattern[0]==target[0] or pattern[0]=='.'):
        return match(target[1:], pattern[1:])
    
    return False

# print(match('aaa', 'a.a'))
# print(match('aaa', 'ab*ac*a'))
# print(match('',''))
# print(match('aaa', '...'))
# print(match('', 'a*b*c*'))

# print(match('aaa', 'aa.a'))
# print(match('aaa', 'ab*a'))
    

'''
面试题20：表示数值的字符串
题目：请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串'+100'、'5e2'、'-123'、'3.1415'及'-1E-16'都表示数值，但'12e'、'1a3.14'、'1.2.3'、'+-5'及'12e+5.4'都不是。
'''
'''
数字模式：A[.[B]][e|EC]或.B[e|EC]，其中[]表示可有可无，A和C是可带正负号的整数，B是无符号整数
注意：e或E前必须有数字，如1e-5或1.e-5
注意：python中数字和字符串都是值传递
牛客网check
'''
def isNumeric(string):
    if string == '':
        return False

    # 判断是否有A部分，有则True，无则False，返回的string是去掉A部分的
    numeric, string = isInteger(string) 

    # 判断是否有B部分
    if string != '' and string[0] == '.':
        string = string[1:]
        B, string = isUnsignedInteger(string)
        numeric = B or numeric
    
    # 判断是否有C部分，注意e或E前必须有数字，故用and
    if string != '' and (string[0] == 'e' or string[0] == 'E'):
        string = string[1:]
        C, string = isInteger(string)
        numeric = C and numeric
    
    return (numeric and string == '')

def isUnsignedInteger(string):
    if string == '':
        return False, string

    n = len(string)
    while(string != '' and string[0] >= '0' and string[0] <= '9'):
        string = string[1:]
    
    return (n > len(string)), string

def isInteger(string):
    if string == '':
        return False, string
    
    if string[0] == '+' or string[0] == '-':
        string = string[1:]
    return isUnsignedInteger(string)

# print(isNumeric('123'))
# print(isNumeric('123.'))
# print(isNumeric('.5'))
# print(isNumeric('123.5e-3'))
# print(isNumeric(''))
# print(isNumeric('123.5e'))
# print(isNumeric('1a.32'))


'''
面试题21：调整数组顺序使奇数位于偶数前面
题目：输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分
可扩展的解法（isOdd）
牛客网check（不改变子集的顺序）
'''
def Reorder(arr):
    if len(arr) <= 1:
        return arr

    p1 = 0
    p2 = len(arr) - 1
    
    while (p1 < p2):
        while (p1 < p2 and isOdd(arr[p1])):
            p1 += 1
        while (p1 < p2 and not isOdd(arr[p2])):
            p2 -= 1
        
        if p1 < p2:
            temp = arr[p1]
            arr[p1] = arr[p2]
            arr[p2] = temp
    return arr

from collections import deque
# 不改变子集的顺序（空间复杂度）
def Reorder2(arr):
    if len(arr) <= 1:
        return arr

    # 要放后面的从前往后append
    # 要放前面的从后往前appendleft
    new_arr = deque()
    for i in range(len(arr)):
        if not isOdd(arr[i]):
            new_arr.append(arr[i])
        if isOdd(arr[len(arr) - i - 1]):
            new_arr.appendleft(arr[len(arr) - i - 1])
    return list(new_arr)


def isOdd(num):
    return num & 0x1

# 负数在非负数前面
def isNegative(num):
    return (num < 0)

# 能被3整除的数在前面
def isDividedBy3(num):
    return (num % 3 == 0)

print(Reorder2([1,3,5,2,4,6]))
print(Reorder2([1,2,3,4,5,6,7]))
print(Reorder2([2,4,6,1,3,5]))
print(Reorder2([1,3,5]))
print(Reorder2([2,4,6]))
print(Reorder2([]))


'''
面试题22：链表中倒数第k个节点
题目：输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个节点是值为4的节点。
牛客网check
'''
def KthNodeToTail(pHead, k):
    if pHead is None or k <= 0:
        return False
    
    pFast = pHead
    pSlow = pHead
    for i in range(0, k-1):
        if pFast.next is not None:
            pFast = pFast.next 
        else:
            return False # important, in case k is larger than link length
    
    while(pFast.next is not None):
        pFast = pFast.next
        pSlow = pSlow.next
    return pSlow.value

def CenterNode(pHead):
    if pHead is None:
        return False
    
    pFast = pHead
    pSlow = pHead
    while(pFast.next is not None and pFast.next.next is not None):
        pFast = pFast.next.next
        pSlow = pSlow.next
    return pSlow.value

# one, two, three, four = LinkNode(1), LinkNode(2), LinkNode(3), LinkNode(4)
# one.next, two.next, three.next = two, three, four
# print(KthNodeToTail(one, 1))
# print(KthNodeToTail(one, 2))
# print(KthNodeToTail(one, 3))
# print(KthNodeToTail(one, 4))
# print(KthNodeToTail(one, 5))
# print(KthNodeToTail(None, 5))

# print(CenterNode(one))
# print(CenterNode(two))
# print(CenterNode(three))
# print(CenterNode(four))


'''
面试题23：链表中环的入口节点
题目：如果一个链表中包含环，找出环的入口点
牛客网check
'''
def EntryNodeOfLoop(pHead):
    if pHead is None:
        return False 

    loop_n = LoopLength(pHead)
    if not loop_n:
        return False

    pFast = pHead
    pSlow = pHead
    for i in range(0, loop_n):
        pFast = pFast.next 
    
    while(pFast != pSlow):
        pFast = pFast.next
        pSlow = pSlow.next
    return pSlow.value

def LoopLength(pHead):
    if pHead is None:
        return None

    pSlow = pHead
    pFast = pHead.next
    while(pFast is not None and pFast.next is not None and pFast.next.next is not None and pFast != pSlow):
        pFast = pFast.next.next
        pSlow = pSlow.next
    if pFast is pSlow:
        count = 1
        pNext = pFast
        while(pNext.next != pFast):
            pNext = pNext.next
            count += 1
        return count
    else:
        return 0

# one, two, three, four = LinkNode(1), LinkNode(2), LinkNode(3), LinkNode(4)
# one.next, two.next, three.next = two, three, four
# print(EntryNodeOfLoop(one))
# four.next = four
# print(EntryNodeOfLoop(one))
# four.next = three
# print(EntryNodeOfLoop(one))
# four.next = one
# print(EntryNodeOfLoop(one))
# print(EntryNodeOfLoop(four))
# print(EntryNodeOfLoop(None))



'''
面试题24：反转链表
题目：定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。
牛客网check
'''
def ReverseList(pHead):
    if pHead is None:
        return False 
    
    first = None 
    second = pHead
    if pHead.next is None:
        return pHead 
    else:
        third = pHead.next
    
    while(second is not None and third is not None):
        second.next = first
        first, second, third = second, third, third.next
    if second is not None:
        second.next = first
    return second

one, two, three, four = LinkNode(1), LinkNode(2), LinkNode(3), LinkNode(4)
one.next, two.next, three.next = two, three, four
one_r = ReverseList(one)
four_r = ReverseList(four)


'''
面试题25：合并两个排序的链表
题目：输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然使递增排序的。
牛客网check
'''
def Merge(pHead1, pHead2):
    if pHead1 == None:
        return pHead2
    elif pHead2 == None:
        return pHead1
    
    if pHead1.value < pHead2.value:
        pMergeHead = pHead1
        pMergeHead.next = Merge(pHead1.next, pHead2)
    else:
        pMergeHead = pHead2
        pMergeHead.next = Merge(pHead2.next, pHead1)
    return pMergeHead



'''
面试题26：树的子结构
题目：输入两棵二叉树A和B，判断B是不是A的子结构。
牛客网check
'''
def HasSubtree(pRoot1, pRoot2):
    result = False

    if (pRoot1 is not None and pRoot2 is not None):

        result = DoesTree1HasTree2(pRoot1, pRoot2)

        if not result:
            result = HasSubtree(pRoot1.left, pRoot2)
        if not result:
            result = HasSubtree(pRoot1.right, pRoot2)
    
    return result


def DoesTree1HasTree2(pRoot1, pRoot2):
    if pRoot2 is None:
        return True
    if pRoot1 is None:
        return False
    
    if not Equal(pRoot1.value, pRoot2.value):
        return False
    
    return DoesTree1HasTree2(pRoot1.left, pRoot2.left) and \
           DoesTree1HasTree2(pRoot1.right, pRoot2.right)

def Equal(num1, num2):
    if abs(num1 - num2) < 1e-8:
        return True
    else:
        return False





        