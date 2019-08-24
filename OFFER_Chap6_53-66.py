class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

'''
面试题53：在排序数组种查找数字
题目一：数字在排序数组种出现的次数。
统计一个数字在排序数组中出现的次数。例如，输入排序数组[1,2,3,3,3,3,4,5]和数字3，由于3在这个数组中出现了4次，因此输出4。
'''
def GetNumberOfK1(arr, num):
    if arr == []:
        return 0
    index = FindNum(arr, num, 0, len(arr)-1)
    if index is None:
        return 0
    else:
        count = 1
    i = index - 1
    j = index + 1
    while(i >= 0 and arr[i] == num):
        count += 1
        i -= 1
    while(j <= len(arr)-1 and arr[j] == num):
        count += 1
        j += 1
    return count


def FindNum(arr, num):
    if arr == [] :
        return None
    
    start = 0
    end = len(arr) - 1
    while(start <= end):
        mid = (start + end) // 2
        if arr[mid] > num:
            end = mid - 1
        elif arr[mid] < num:
            start = mid + 1
        else:
            return mid

    return None
    
    

def GetNumberOfK2(arr, num):
    if arr == []:
        return 0
    
    first = GetFirstK(arr, num)
    last = GetLastK(arr, num)
    count = 0
    if first > -1 and last > -1:
        count = last - first + 1
    return count

def GetFirstK(arr, num):
    if arr == []:
        return -1
    start = 0
    end = len(arr) - 1
    while(start <= end):
        mid = (start + end) // 2
        if arr[mid] > num:
            end = mid - 1
        elif arr[mid] < num:
            start = mid + 1
        elif arr[mid] == num and (mid == 0 or arr[mid-1] != num): 
            return mid
        else:
            end = mid - 1
    return -1

def GetLastK(arr, num):
    if arr == []:
        return -1
    start = 0
    end = len(arr) - 1
    while(start <= end):
        mid = (start + end) // 2
        if arr[mid] > num:
            end = mid - 1
        elif arr[mid] < num:
            start = mid + 1
        elif arr[mid] == num and (mid == len(arr)-1 or arr[mid+1] != num ): 
            return mid
        else:
            start = mid + 1
    return -1

# print(GetNumberOfK2([1,2,3,3,3,3], 3))
# print(GetNumberOfK2([1,2,3,3,3,3,4,5], 1))
# print(GetNumberOfK2([1,2,3,4,5], 1))
# print(GetNumberOfK2([1,2,3,4,5], 0))

'''
题目二：0~n-1中缺失的数字
一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且范围都在0~n-1之内。在0~n-1内的n个数字中有且只有一个数字不在该数组中，找出这个数字。
'''
def GetMissingNumber(arr):
    if arr == []:
        return -1
    start = 0
    end = len(arr) - 1
    while start <= end:
        mid = (start + end) // 2
        if arr[mid] == mid:
            start = mid + 1
        elif mid == 0 or arr[mid-1] == (mid-1):
            return mid
        else:
            end = mid - 1

    return -1

# print(GetMissingNumber([1,2,3,4]))
# print(GetMissingNumber([0,1,3,4]))
# print(GetMissingNumber([0,1,2,3]))

'''
题目三：数组中数值和下表相等的元素。
假设一个单调递增的数组里的每个元素都是整数并且是唯一的。请编程实现一个函数，找出数组中任意一个数值等于其下标的元素。例如，在数组[-3,-1,1,3,5]中，数值3和它的下标相等。
'''
def GetNumberSameAsIndex(arr):
    if arr == []:
        return -1
    start = 0
    end = len(arr) - 1
    while(start <= end):
        mid = (start + end) // 2
        if arr[mid] == mid:
            return mid
        elif arr[mid] > mid:
            end = mid - 1
        else:
            start = mid + 1
    return -1

# print(GetNumberSameAsIndex([-3,-1,1,3,5]))
# print(GetNumberSameAsIndex([-3,-1,0,1,2]))
# print(GetNumberSameAsIndex([-3,-1]))
# print(GetNumberSameAsIndex([0,1,2,3]))
# print(GetNumberSameAsIndex([0,2,3]))


'''
面试题54：二叉搜索树的第k节点
题目：给定一棵二叉搜索树，请找出其中的第k的结点。例如，（5，3，7，2，4，6，8）中，按结点数值大小顺序第三小结点的值为4。
'''
def KthNode1(pTree, k):
    if pTree is None or k <= 0:
        return None
    res = []
    MidTraverse(pTree, res)
    if k > len(res):
        return None
    return res[k-1]

def MidTraverse(pTree, res):
    if pTree is None:
        return res

    if pTree.left is not None:
        MidTraverse(pTree.left)
    res.append(pTree)
    if pTree.right is not None:
        MidTraverse(pTree.right)
    
    return



def KthNode2(pTree, k):
    if pTree is None or k <= 0:
        return None
    
    target, k = KthNode2_core(pTree, k)
    return target
    
# 注意递归时k要正确传出
def KthNode2_core(pTree, k):
    target = None
    if pTree.left is not None:
        target, k = KthNode2_core(pTree.left, k)
    
    if target is None:
        if k == 1:
            target = pTree
        k -= 1
    
    if target is None and pTree.right is not None:
        target, k = KthNode2_core(pTree.right, k)
    
    return target, k


# a, b, c, d, e, f, g = TreeNode(8), TreeNode(6), TreeNode(10), TreeNode(5), TreeNode(7), TreeNode(9), TreeNode(11) 
# a.left, a.right, b.left, b.right, c.left, c.right = b, c, d, e, f, g
# target = KthNode2(a, 1)
# target = KthNode2(a, 2)
# print(target.val)


'''
面试题55：二叉树的深度。
题目一：二叉树的深度。
输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
'''
def TreeDepth(pRoot):
    if pRoot is None:
        return 0
    
    left = TreeDepth(pRoot.left)
    right = TreeDepth(pRoot.right)

    return (left + 1) if left > right else (right + 1)

'''
题目二：平衡二叉树。
输入一棵二叉树的根节点，判断该数是不是平衡二叉树。如果某二叉树中任意节点的左、右子树的深度相差不超过1，就是一棵平衡二叉树。
'''
# 节点重复遍历
def IsBalancedTree1(pRoot):
    if pRoot is None:
        return True

    left = TreeDepth(pRoot.left)
    right = TreeDepth(pRoot.right)

    if abs(left-right) > 1:
        return False
    
    return IsBalancedTree1(pRoot.left) and IsBalancedTree1(pRoot.right)

# 节点只遍历一次
def IsBalancedTree2(pRoot):
    if pRoot is None:
        depth = 0
        return (True, depth)

    left = IsBalancedTree2(pRoot.left)
    right = IsBalancedTree2(pRoot.right)
    depth = (1 + left[1]) if left[1] > right[1] else (1 + right[1])
    
    if (left[0] and right[0]):
        if abs(left[1] - right[1]) <= 1:
            return (True, depth)
    
    return (False, depth)
            
    
# a, b, c, d, e = TreeNode(1), TreeNode(2), TreeNode(3), TreeNode(4), TreeNode(5)  
# a.left, a.right, b.left, b.right = b, c, d, e
# print(IsBalancedTree2(a))
# print(IsBalancedTree2(c))
# a.left, a.right, b.left, b.right = b, None, d, None
# print(IsBalancedTree2(a))
# a.left, a.right, b.left, b.right, c.right = None, c, None, None, b
# print(IsBalancedTree2(a))



'''
面试题56：数组中数字出现的次数。
题目一：数组中只出现一次的两个数字。
一个整型数组里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度O(n)，空间复杂度O(1)。
'''
def FindNumsAppearOnce(arr):
    if arr == []:
        return arr 
    
    xor = 0
    for num in arr:
        xor ^= num
    
    index = IdxOfFirstOne(xor)

    res1 = 0
    res2 = 0
    for num in arr:
        if IsIndexOne(num, index):
            res1 ^= num
        else:
            res2 ^= num
    return res1, res2

def IdxOfFirstOne(num):
    index = 0
    while(num & 1 == 0):
        num = num >> 1
        index += 1
    return index

def IsIndexOne(num, index):
    num = num >> index
    return (num & 1 == 1)


# print(FindNumsAppearOnce([1,2,2,4,4,3]))
# print(FindNumsAppearOnce([1,3]))

'''
题目二：数组中唯一只出现一次的数字。
在一个数组中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。
'''
def FindNumAppearOnce(arr):
    if arr == []:
        return None
    
    bits = [0] * 32
    for num in arr:
        mask = 1
        for j in range(31, -1, -1):
            if (num & mask == 1):
                bits[j] += 1
                mask = mask << 1
    
    res = 0
    for num in bits:
        res = res << 1
        res += num % 3
    return res

# print(FindNumAppearOnce([1,2,2,2]))
# print(FindNumAppearOnce([1,2,2,2,3,3,3]))
# print(FindNumAppearOnce([0]))



'''
面试题57：和为s的数字
题目一：和为s的两个数字。
输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的/任意一对。
牛客网check
'''
def FindNumbersWithSum(arr, num):
    if arr == []:
        return None
    start = 0
    end = len(arr) - 1
    while (start < end):
        if arr[start] + arr[end] == num:
            return (arr[start], arr[end])
        elif arr[start] + arr[end] < num:
            start += 1
        else:
            end -= 1
    
    return None

# print(FindNumbersWithSum([1,2,4,7,11,15], 15))
# print(FindNumbersWithSum([1,2,3,4,5,6], 7))

'''
题目二：和为s的连续正树序列。
输入一个正数s，打印出所有和为s的连续正数序列（至少含有两个数）。例如，输入15，打印出1~5，4~6和7~8。
牛客网check
'''
def FindContinuousSequence(sum):
    if sum <= 2:
        return []
    small = 1
    big = 2
    middle = sum // 2
    curSum = small + big
    res = []
    while(small <= middle):
        if curSum == sum:
            res.append(list(range(small, big+1)))
        while curSum > sum and small <= middle:
            curSum -= small
            small += 1

            if curSum == sum:
                res.append(list(range(small, big+1)))
        big += 1
        curSum += big
    return res

# print(FindContinuousSequence(15))
# print(FindContinuousSequence(1))
# print(FindContinuousSequence(3))



'''
面试题58：翻转字符串。
题目一：翻转单词顺序。
输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串是"I am a student."，则输出"student. a am I"。
牛客网check
'''
def ReverseSentence(string):
    if string == '':
        return ''
    if isinstance(string, str):
        string = list(string)
    
    string = Reverse(string, 0, len(string)-1)
    start = end = 0
    while(start < len(string)):

        # if string[start] == ' ':
        #     start += 1
        #     # end += 1
        if end == len(string) or string[end] == ' ':
            string = Reverse(string, start, end-1)
            start = end + 1
        end += 1
    return ''.join(string)

def Reverse(string, start, end):
    if start < 0 and end >= len(string) and start >= end:
        return string 

    while(start < end):
        temp = string[start]
        string[start] = string[end]
        string[end] = temp
        start += 1
        end -= 1
    return string

# print(ReverseSentence("I am a student."))


'''
题目二：左旋转字符串。
字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。例如，输入字符串"abcdefg"和数字2，返回左旋转两位得到的结果"cdefgab"。
'''
def LeftRotateString1(string, n):
    if string == '' or n <= 0 or n > len(string):
        return string
    tail = string[:n]
    return string[n:] + tail
print(LeftRotateString('abcdefg', 2))


def LeftRotateString2(string, n):
    if string == '' or n <= 0 or n > len(string):
        return string
    if isinstance(string, str):
        string = list(string)

    string = Reverse(string, 0, n-1)
    string = Reverse(string, n, len(string)-1)
    string = Reverse(string, 0, len(string)-1)
    return ''.join(string)

        
