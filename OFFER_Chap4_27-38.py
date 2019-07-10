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

a, b, c, d, e = TreeNode(1), TreeNode(2), TreeNode(3), TreeNode(4), TreeNode(5)
a.right, b.right= b, c
PrintTreeMultiLines(a)


'''
题目三：之字形打印二叉树
'''
def PrintTreeShapeZ(pTree):
    pass

    