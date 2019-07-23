class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

'''
面试题27：二叉树的镜像
题目：请完成一个函数，输入一棵二叉树，输出它的镜像。
牛客网check
'''
# 递归法
def MirrorRecursively(pRoot):
    if pRoot is None:
        return None
    if pRoot.left is None and pRoot.right is None:
        return None
    
    temp = pRoot.left
    pRoot.left = pRoot.right
    pRoot.right = temp

    if pRoot.left is not None:
        MirrorRecursively(pRoot.left)
    if pRoot.right is not None:
        MirrorRecursively(pRoot.right)


# 循环法
def Mirror(pRoot):
    if pRoot is None:
        return None

    nodes = []
    nodes.append(pRoot)

    while(nodes != []):
        pcur = nodes.pop(0)
        if pcur.left is not None:
            nodes.append(pcur.left)
        if pcur.right is not None:
            nodes.append(pcur.right)

        if pcur.left is not None or pcur.right is not None:
            temp = pcur.left
            pcur.left = pcur.right
            pcur.right = temp

# a, b, c = TreeNode(1), TreeNode(2), TreeNode(3)
# a.left, b.left= b, c 
# Mirror(a)
# print(a)


'''
面试题28：对称的二叉树
题目：请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。
牛客网check（两种）
'''
from copy import deepcopy
def isSymmetrical(pRoot):
    if pRoot is None:
        return True

    pRoot_cp = deepcopy(pRoot)
    MirrorRecursively(pRoot_cp)

    return isIdentical(pRoot, pRoot_cp)


def isIdentical(pRoot1, pRoot2):
    if pRoot1 is None and pRoot2 is None:
        return True
    if pRoot1 is None or pRoot2 is None:
        return False
    
    if pRoot1.val != pRoot2.val:
        return False
    
    return isIdentical(pRoot1.left, pRoot2.left) and isIdentical(pRoot1.right, pRoot2.right)


def isSymmetrical2(pRoot):
    if pRoot is None:
        return True
    
    return isSymmetrical2_core(pRoot, pRoot)

def isSymmetrical2_core(pRoot1, pRoot2):
    if pRoot1 is None and pRoot2 is None:
        return True
    if pRoot1 is None or pRoot2 is None:
        return False

    if pRoot1.val != pRoot2.val:
        return False
    
    return isSymmetrical2_core(pRoot1.left, pRoot2.right) and isSymmetrical2_core(pRoot1.right, pRoot2.left)


# a, b, c = TreeNode(1), TreeNode(3), TreeNode(2)
# a.left, a.right= b, c 
# res = isSymmetrical2(a)
# print(res)



'''
面试题29：顺时针打印矩阵
题目：输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。例如输入如下矩阵：
1  2  3  4
5  6  7  8
9  10 11 12
13 14 15 16
则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10
牛客网check
'''
from itertools import chain 
def PrintMatrixClockwise(mat):
    rows = len(mat)
    cols = len(mat[0])
    start = 0
    res = []


    if rows <= 1 or cols <= 1:
        return list(chain.from_iterable(mat))

    for start in range((min(rows, cols) + 1)//2):
        res += PrintMatrix(mat, rows, cols, start)
    return res

    
def PrintMatrix(mat, rows, cols, start):
    endX = rows - 1 - start 
    endY = cols - 1 - start
    res = []

    # 打印上一行
    for i in range(start, endY+1):
        res.append(mat[start][i])
    
    # 打印右一列
    # 至少两行endX-start+1>1
    if start < endX: 
        for i in range(start+1, endX+1):
            res.append(mat[i][endY])
    
    # 打印下一行 
    # 至少两行两列endX-start+1>1 && endY-start-1>1
    if start < endX and start < endY: 
        for i in range(endY-1, start-1, -1):
            res.append(mat[endX][i])
    
    # 打印左一列 
    # 至少三行两列endX-start+1>2 && endY-start-1>1
    if start < endX - 1 and start < endY: 
        for i in range(endX-1, start, -1):
            res.append(mat[i][start])
    return res


# mat1 = [[1,2,3], [4,5,6], [7,8,9]] # 一步
# mat2 = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
# mat3 = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]] # 三步
# mat4 = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]]
# mat = [[1]]
# PrintMatrixClockwise(mat1)
# PrintMatrixClockwise(mat2)
# PrintMatrixClockwise(mat3)
# PrintMatrixClockwise(mat4)
# print(PrintMatrixClockwise(mat))

    


'''
面试题30：包含min函数的栈
题目：定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的min函数。在该栈中，调用min、push及pop的时间复杂度都是O(1)。
牛客网check
'''
class Stack(object):
    def __init__(self):
        self.m_data = []
        self.m_min = []
    
    def push(self, value):
        assert len(self.m_data) == len(self.m_min)
        self.m_data.append(value)
        if len(self.m_min) <= 0 or value < self.m_min[-1]:
            self.m_min.append(value)
        else:
            self.m_min.append(self.m_min[-1])
    
    def pop(self):
        assert len(self.m_data) > 0
        self.m_data.pop()
        self.m_min.pop()
    
    def min(self):
        assert len(self.m_data) > 0
        return self.m_min[-1]

# stack = Stack()
# stack.push(2), stack.push(4), stack.push(3), stack.push(1),
# print(stack.m_data)
# print(stack.min())
# stack.pop()
# print(stack.m_data)
# print(stack.min())
# stack.pop()
# print(stack.m_data)
# print(stack.min())
# stack.pop()
# print(stack.m_data)
# print(stack.min())
# stack.pop()
# print(stack.m_data)

        


'''
面试题31：栈的压入、弹出序列
题目：输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列[1,2,3,4,5]是某栈的压栈序列，序列[4,5,3,2,1]是该压栈序列对应的一个弹出序列，但[4,3,5,1,2]就不可能是该压栈序列的弹出序列。
牛客网check
'''
def isPopOrder(push, pop):
    assert len(push) == len(pop)
    if push == [] or pop == []:
        return False
    
    stack = []
    while(pop != []):
        pop_v = pop.pop(0)
        while (stack == [] or stack[-1] != pop_v) and len(push) > 0 :
            stack.append(push.pop(0))
        if stack[-1] == pop_v:
            stack.pop()
    if stack == []:
        return True
    else:
        return False

# print(isPopOrder([1,2,3,4,5], [4,5,3,2,1]))
# print(isPopOrder([1,2,3,4,5], [4,5,3,1,2]))

    

'''
面试题32：从上到下打印二叉树
题目一：不分行从上到下打印二叉树
从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
牛客网check
'''
def PrintTreeOneLine(pTree):
    nodes = []
    result = []
    if pTree is None:
        return []
    else:
        nodes.append(pTree)

    while(len(nodes) > 0):
        pNow = nodes.pop(0)
        result.append(pNow.val)
        if pNow.left is not None:
            nodes.append(pNow.left)
        if pNow.right is not None:
            nodes.append(pNow.right)
    return result



'''
题目二：分行从上到下打印二叉树
从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印一行。
'''
def PrintTreeMultiLines(pTree):
    nodes = []
    if pTree is None:
        return None
    else:
        nodes.append(pTree)
        toBePrinted = 1
        nextLevel = 0
    while(nodes != []):
        pNow = nodes.pop(0)
        print(pNow.val, end = ' ')
        toBePrinted -= 1
        if pNow.left is not None:
            nodes.append(pNow.left)
            nextLevel += 1
        if pNow.right is not None:
            nodes.append(pNow.right)
            nextLevel += 1
        if toBePrinted == 0:
            print('\n')
            toBePrinted = nextLevel
            nextLevel = 0

# a, b, c, d, e = TreeNode(1), TreeNode(2), TreeNode(3), TreeNode(4), TreeNode(5)
# a.right, b.right= b, c
# PrintTreeMultiLines(a)


'''
题目三：之字形打印二叉树
第一行从左到右，第二行从右到左，以此类推。
'''
def PrintTreeShapeZ(pTree):
    if pTree is None:
        return None 
    stack1 = []
    stack2 = []
    order = 1 # 1时先存左子节点再右子节点，0时反之
    stack1.append(pTree)
    while(stack1 != [] or stack2 != []):
        if order:
            pNode = stack1.pop()
            print(pNode.val, end=' ')
            if pNode.left is not None:
                stack2.append(pNode.left)
            if pNode.right is not None:
                stack2.append(pNode.right)
            if stack1 == []:
                print('\n')
                order = 1 - order
        else:
            pNode = stack2.pop()
            print(pNode.val, end=' ')
            if pNode.right is not None:
                stack1.append(pNode.right)
            if pNode.left is not None:
                stack1.append(pNode.left)
            if stack2 == []:
                print('\n')
                order = 1 - order

    
# a, b, c, d, e = TreeNode(1), TreeNode(2), TreeNode(3), TreeNode(4), TreeNode(5)
# a.left, b.left  = b, d
# a.right, b.right = c, e
# PrintTreeShapeZ(a)


    

'''
面试题33：二叉搜索树的后序遍历序列
题目：输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。是则返回True，否则返回False。假设输入数组的任意两个数字都互不相同。
牛客网check
'''
def VerifySquenceOfBST(arr):
    n = len(arr)
    if n == 0:
        return False
    
    root = arr[-1]
    for i in range(n):
        if arr[i] > root:
            break
    for j in range(i, n):
        if arr[j] < root:
            return False

    left = True
    if i > 0:
        left = VerifySquenceOfBST(arr[:i])

    right = True
    if i < n - 1:
        right = VerifySquenceOfBST(arr[i:(n-1)])
    
    return (left and right)
            
# print(VerifySquenceOfBST([4,8,6,12,16,14,10]))            
# print(VerifySquenceOfBST([4,6,7,5]))            



'''
面试题34：二叉树中和为某一值的路径
题目：输入一棵二叉树和整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。
牛客网check
'''
from copy import deepcopy
def FindPath(pTree, num):
    if pTree is None:
        return [] 
    
    curSum = 0
    curPath = []
    res = []
    FindPathCore(pTree, num, curSum, curPath, res)
    return res

def FindPathCore(pNode, num, curSum, curPath, res):
    curSum += pNode.val
    curPath.append(pNode.val)

    isleaf = pNode.left is None and pNode.right is None
    if isleaf and curSum == num:
        res.append(deepcopy(curPath))

    if pNode.left is not None:
        FindPathCore(pNode.left, num, curSum, curPath, res)
    if pNode.right is not None:
        FindPathCore(pNode.right , num, curSum, curPath, res)

    curPath.pop()
    curSum -= pNode.val


# a, b, c, d, e = TreeNode(1), TreeNode(2), TreeNode(3), TreeNode(4), TreeNode(5)
# a.left, b.left  = b, d
# a.right, b.right = c, e
# print(FindPath(a, 1))



'''
面试题35：复杂链表的复制
题目：请实现函数复制一个复杂链表。在复杂链表中，每个节点除了有一个pNext指针指向下一个节点，还有一个pSibling指针指向链表中的任意节点或者None。复杂链表节点的定义如下：
牛客网check(2种方法)
'''
class ComplexLinkNode(object):
    def __init__(self, x):
        self.label = x 
        self.next = None
        self.random = None
    
def CloneHash(pTree):
    if pTree is None:
        return None

    hash = {}
    pNode = pTree

    while(pNode is not None):
        pClone = ComplexLinkNode(pNode.label)
        hash[pNode] = pClone
        pNode = pNode.next
    
    for k, k_clone in hash.items():
        if k.next is not None:
            k_clone.next = hash[k.next]
        if k.random is not None:
            k_clone.random = hash[k.random]
    
    return hash[pTree]

# node1, node2, node3, node4 = ComplexLinkNode(1), ComplexLinkNode(2), ComplexLinkNode(3), ComplexLinkNode(4)
# node1.next, node2.next, node3.next = node2, node3, node4
# pCloned = CloneHash(node1)
# print(pCloned.label)

def Clone(pTree):
    if pTree is None:
        return None

    CloneNodes(pTree)
    CloneSiblingNodes(pTree)
    return Reconstruct(pTree)

def CloneNodes(pTree):
    pNode = pTree
    while(pNode is not None):
        pClone = ComplexLinkNode(pNode.label)
        pClone.next = pNode.next
        pNode.next = pClone
        pNode = pClone.next

def CloneSiblingNodes(pTree):
    pNode = pTree
    while(pNode is not None):
        if pNode.random is not None:
            pNode.next.random = pNode.random.next
        pNode = pNode.next.next

def Reconstruct(pTree):
    pClonedHead = pTree.next
    pTree.next = pClonedHead.next

    pClonedNode = pClonedHead
    pNode = pClonedNode.next

    while(pNode is not None):
        pClonedNode.next = pNode.next
        pClonedNode = pClonedNode.next
        pNode.next = pClonedNode.next
        pNode = pNode.next
    
    return pClonedHead


# node1, node2, node3, node4 = ComplexLinkNode(1), ComplexLinkNode(2), ComplexLinkNode(3), ComplexLinkNode(4)
# node1.next, node2.next, node3.next = node2, node3, node4
# pCloned = Clone(node1)
# print(pCloned.label)



'''
面试题36：二叉搜索树与双向链表
题目：输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。
'''
def Convert(pHeadOfTree):
    if pHeadOfTree is None:
        return None

    pLastNodeInList = ConvertNode(pHeadOfTree, None)

    # 找到双向链表的头
    pHeadOfList = pLastNodeInList
    while(pHeadOfList is not None and pHeadOfList.left is not None):
        pHeadOfList = pHeadOfList.left
    print(pHeadOfList is pLastNodeInList)
    
    return pHeadOfList 

def ConvertNode(pNode, pLastNodeInList):
    if pNode is None:
        return 
    
    if pNode.left is not None:
        pLastNodeInList = ConvertNode(pNode.left, pLastNodeInList)
    
    pNode.left = pLastNodeInList
    if pLastNodeInList is not None:
        pLastNodeInList.right = pNode
    
    pLastNodeInList = pNode
    if pNode.right is not None:
        pLastNodeInList = ConvertNode(pNode.right, pLastNodeInList)

    return pLastNodeInList
    
# a, b, c, d, e = TreeNode(1), TreeNode(2), TreeNode(3), TreeNode(4), TreeNode(5)
# a.left, b.left  = b, d
# a.right, b.right = c, e
# DoubleLink = Convert(a)

    


'''
面试题37：序列化二叉树
题目：请实现两个函数，分别用来序列化和反序列化二叉树。
'''
def Serialize(pRoot):
    res = ''
    if pRoot is None:
        res += '$,'
        return res

    res = res + str(pRoot.val) + ','
    res += Serialize(pRoot.left)
    res += Serialize(pRoot.right)

    return res

def Deserialize(s):
    if isinstance(s, str):
        s = s.split(',')

    if s == [] :
        return None
    if s[0] == '$':
        s.pop(0)
        return None

    pRoot = TreeNode(int(s[0]))
    s.pop(0)
    pRoot.left = Deserialize(s)
    pRoot.right = Deserialize(s)

    return pRoot


# a, b, c, d, e = TreeNode(1), TreeNode(2), TreeNode(3), TreeNode(4), TreeNode(5)
# a.left, b.left  = b, d
# a.right, b.right = c, e
# print(Serialize(a))
# root = Deserialize(Serialize(a))
# print(root.val)


'''
面试题38：字符串的排列
题目：输入一个字符串，打印出该字符串中字符的所有排列。例如，输入字符串abc，则打印出由字符a，b，c所能排列出来的所有字符串abc、acb、bac、bca、cab和cba。
'''
def Permutation(s):
    if isinstance(s, str):
        s = list(s)
    if len(s) == 0 or len(s) == 1:
        return s
    res = []
    Permutation_core(s, 0, res) 
    return res

def Permutation_core(s, begin, res):
    if begin == len(s) - 1:
        string = ''.join(s)
        if string not in res:
            res.append(string)
        return 
        
    for i in range(begin, len(s)):
        temp = s[begin]
        s[begin] = s[i]
        s[i] = temp
        Permutation_core(s, begin+1, res)
        temp = s[begin]
        s[begin] = s[i]
        s[i] = temp
    return 
        
# print(Permutation('aab'))


'''
扩展：求字符的所有组合。如字符abc，它们的组合有a、b、c、ab、ac、bc、abc。
'''
def Combination(s):
    if s == '':
        return []
    
    res = []
    for i in range(1, len(s)+1):
        Combination_core(s, '', i, 0, res)
    return res

def Combination_core(s, word, length, begin, res):
    if len(word) == length:
        res.append(word)
        return

    for i in range(begin, len(s)):
        word += s[i]
        Combination_core(s, word, length, i+1, res)
        word = word.strip(s[i]) # 回溯
        
# print(Combination('abc'))



'''
扩展：输入一个含有8个数字的数组，判断有没有可能把这8个数字分别放到正方体的8个顶点上，使得正方体上三组相对的面上的4个顶点的和都相等。
'''
from copy import copy
def Square(arr):
    if len(arr) != 8 :
        return []
    
    res = []
    Square_core(arr, 0, res)
    return res

def Square_core(arr, begin, res):
    if begin == len(arr) - 1:
        if arr[0]+arr[1]+arr[2]+arr[3]==arr[4]+arr[5]+arr[6]+arr[7] \
           and arr[0]+arr[1]+arr[4]+arr[5]==arr[2]+arr[3]+arr[6]+arr[7] \
           and arr[0]+arr[3]+arr[4]+arr[7]==arr[1]+arr[2]+arr[5]+arr[6] :
           if arr not in res:
                res.append(copy(arr)) # 注意copy
        return 

    for i in range(begin, len(arr)):
        temp = arr[begin]
        arr[begin] = arr[i]
        arr[i] = temp
        Square_core(arr, begin+1, res)
        temp = arr[begin]
        arr[begin] = arr[i]
        arr[i] = temp
    

print(Square([1,1,1,1,1,1,1,1]))
print(Square([1,1,1,1,2,2,2,2]))


'''
扩展：八皇后。n*n的国际象棋上摆放n个皇后，使其不在同一行、同一列或同一对角线。
'''
# 解法一：回溯法，逐个确定皇后的位置
def queen(n):
    if n <= 0:
        return []
    res = []
    arr = [None] * n
    queen_core(n, arr, 0, res)
    return res

def queen_core(n, arr, col, res):
    if col == n:
        res.append(copy(arr))
        return
    for row in range(n):
        arr[col] = row
        flag = True
        for i in range(col):
            if arr[i] == row or abs(arr[i] - arr[col]) == col - i:
                flag = False
                break
        if flag:
            queen_core(n, arr, col+1, res)

print(queen(3))
print(queen(4))
print(len(queen(8)))

# 解法二：全排列，将满足条件的排列方式保存下来
def queen_2(n):
    if n <= 0:
        return []

    arr = list(range(n))
    res = []
    queen_2_core(arr, 0, res)
    return res

def queen_2_core(arr, begin, res):
    n = len(arr)
    if begin == n - 1:
        flag = True
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                if abs(arr[i] - arr[j]) == abs(i-j):
                    flag = False
                    break
        if flag:
            res.append(copy(arr))
        return

    for i in range(begin, n):
        temp = arr[i]
        arr[i] = arr[begin]
        arr[begin] = temp
        queen_2_core(arr, begin+1, res)
        temp = arr[i]
        arr[i] = arr[begin]
        arr[begin] = temp

print(queen_2(3))
print(queen_2(4))
print(len(queen_2(8)))






