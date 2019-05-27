'''
面试题3：数组中重复的数字
在一个长度为 n 的数组中所有数字都在 0~n-1 范围内，找出数组中任意一个重复的数字
'''
def duplicate(nums):
    if len(nums) == 0 :
        return False 
    if max(nums) > len(nums) - 1 or min(nums) < 0:
        return False 
    
    # repeat = []
    for i in range(len(nums)):
        while(nums[i] != i):
            if nums[nums[i]] == nums[i]:
                # repeat.append(nums[i])
                # break
                return nums[i]
            else:
                temp = nums[i]
                nums[i] = nums[temp]
                nums[temp] = temp 
    return False


# print(duplicate([]) == False)
# print(duplicate([1,2,0,0,5]) == False)
# print(duplicate([1,-1,0,0,5]) == False)
# print(duplicate([1,2,3]) == False)
# print(duplicate([1,2,3,2,0,0]) == 2)


'''
在一个长度为 n+1 的数组里所有数字都在 1~n 的范围内，
不修改输入的数组，找出数组中任意一个重复的数字
'''
def getDuplication(nums):
    if len(nums) == 0:
        return -1
    
    start = 1
    end = len(nums) - 1
    while(end >= start):
        # print("start:",start, " end:",end)
        middle = (end + start) // 2 
        # print("middle:", middle)
        count = countRange(nums, len(nums), start, middle)
        # print("count:", count)
        if end == start:
            if count > 1:
                return start
            else:
                break
        
        if count > (middle - start + 1):
            end = middle
        else:
            start = middle + 1
        
    return -1

def countRange(nums, length, start, end):
    if len(nums) == 0:
        return 0
    
    count = 0
    for i in range(length):
        if nums[i] >= start and nums[i] <= end:
            count += 1
    return count

# print(getDuplication([2,3,5,4,6,2,6,7]) == 6)

'''
面试题4：二维数组中的查找
在一个二维数组中，每一行从左到右递增，每一列从上到下递增。请完成一个
函数，输入这样的二维数组和一个整数，判断数组中是否含有该整数
'''
import numpy as np
def Find(mat, rows, cols, num):
    if rows <= 0 or cols <= 0 :
        return False
    row = 0 
    col = cols - 1
    while(row < rows and col >= 0):
        if mat[row][col] == num:
            return True
        elif mat[row][col] > num:
            col -= 1
        else:
            row += 1
    return False 

mat = [[1,2,8,9],
       [2,4,9,12],
       [4,7,10,13],
       [6,8,11,15]]

# print(Find(mat, rows=4, cols=4, num=0) == False)
# print(Find(mat, rows=4, cols=4, num=16) == False)
# print(Find(mat, rows=4, cols=4, num=5) == False)
# print(Find([], rows=0, cols=0, num=5) == False)

# print(Find(mat, rows=4, cols=4, num=1) == True)
# print(Find(mat, rows=4, cols=4, num=15) == True)
# print(Find(mat, rows=4, cols=4, num=10) == True)



'''
面试题5：替换空格
实现一个函数，把字符串中的每个空格替换成 "%20"。如输入 "We are happy."
则输出 "We%20are%20happy."
'''
def replaceBank(string):
    string = list(string)
    length = len(string)
    if length == 0:
        return False
    count = 0
    for i in range(length):
        if string[i] == " ":
            count += 1
    
    newlen = length + count * 2
    string += [None] * count * 2
    p1 = length - 1
    p2 = newlen - 1
    while(p1 >= 0 and p2 > p1):
        if string[p1] == " ":
            string[p2] = "0"
            string[p2-1] = "2"
            string[p2-2] = "%"
            p2 -= 3
        else:
            string[p2] = string[p1]
            p2 -= 1
        p1 -= 1
    return "".join(string)

# print(replaceBank("We are happy.") == "We%20are%20happy.")
# print(replaceBank(" We  are happy ") == "%20We%20%20are%20happy%20")
# print(replaceBank("  ") == "%20%20")
# print(replaceBank("abc") == "abc")
# print(replaceBank("") == False)

'''
-------------------------------------链表-----------------------------------------------
'''
class ListNode:
    def __init__(self, value):
        self.value = value 
        self.next = None

def addToTail(head, value):
    '''
    向链表末尾添加一个节点
    '''
    new = ListNode(value)

    if head is None:
        head = new 
    else:
        temp = head 
        while(temp.next is not None):
            temp = temp.next 
        temp.next = new 
    return head

def printLink(link):
    temp = link 
    while temp is not None:
        print(temp.value)
        temp = temp.next

# link1 = addToTail(None, 100)
# print(link1.value)
# print(link1.next)
# link1 = addToTail(link1, 50)
# link1 = addToTail(link1, 25)
# link1 = addToTail(link1, 12.5)
# link1 = addToTail(link1, 6.25)
# print("Add to tail:")
# printLink(link1)

def removeNode(head, value):
    '''
    在链表中找到第一个含有某值的节点并删除
    '''
    if head is None:
        return False # 链表为空
    else:
        if head.value == value:
            deleted = head 
            head = head.next
        else:
            temp = head 
            while temp.next is not None and temp.next.value != value:
                temp = temp.next 
            if temp.next is not None and temp.next.value == value:
                deleted = temp.next 
                temp.next = temp.next.next 
            else:
                return False # 没有该值
    deleted.next = None
    return head, deleted.value 
                
# print(removeNode(None, 100))
# print(removeNode(link1, 0))
# link1, _= removeNode(link1, 100)
# link1, _ = removeNode(link1, 25)
# link1, _ = removeNode(link1, 6.25)
# print("Remove node:")
# printLink(link1)


'''
面试题6：从尾到头打印链表
输入一个链表的头节点，从尾到头反过来打印出每个节点的值。
链表节点定义如下：
class ListNode:
    def __init__(self, value):
        self.value = value
        self.next = None
'''
def printLinkReverssingly(head):
    stack = []
    if head is None:
        return False
    else:
        temp = head 
        while temp is not None:
            stack.append(temp)
            temp = temp.next
        while len(stack) > 0:
            temp = stack.pop()
            print(temp.value)
# printLinkReverssingly(link1)
# printLinkReverssingly(None)



'''
----------------------------------------树------------------------------------------------
'''
'''
面试题7：重建二叉树
输入某二叉树的前序遍历和中序遍历结果，重建该二叉树，假设输入的前序遍历和中序遍历的结果
中都不包含重复的数字。如：输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，
重建二叉树并输出它的头节点。二叉树节点定义如下：
'''
class BinaryTreeNode:
    def __init__(self, value):
        self.value = value 
        self.left = None 
        self.right = None

def Construct(preorder, inorder):
    assert len(preorder) == len(inorder)
    length = len(preorder)
    if preorder is None or inorder is None or length <= 0:
        return None 
    return ConstructCore(preorder, inorder, length)

def ConstructCore(preorder, inorder, length):
    # 前序遍历的第一个数字是根节点的值
    rootValue = preorder[0]
    root = BinaryTreeNode(rootValue)
    if length == 1 and preorder == inorder:
        return root 
    # 在中序遍历序列中找到根节点的值
    i = 0
    while i < length-1 and inorder[i] != rootValue:
        i += 1
    if i == length-1 and inorder[i] != rootValue:
        raise RuntimeError("Invalid input.")
    leftLength = i
    rightLength = length - i - 1
    leftPre = preorder[1: i+1]
    rightPre = preorder[i+1: ]
    leftIn = inorder[:i]
    rightIn = inorder[i+1: ]
    if leftLength > 0:
        root.left = ConstructCore(leftPre, leftIn, leftLength)
    if rightLength > 0:
        root.right = ConstructCore(rightPre, rightIn, rightLength)
    return root

# tree1 = Construct([1,2,4,7,3,5,6,8], [4,7,2,1,5,3,8,6])
# tree2 = Construct([1], [1])
# tree3 = Construct([], [])
# tree4 = Construct([1,2,4,7,3,5,6,8], [4,7,2,0,5,3,8,6]) # raise error

'''
面试题8：二叉树的下一个节点
给定一棵二叉树和其中的一个节点，如何找出中序遍历序列的下一个节点？树中的节点
除了有两个分别指向左、右子节点的指针，还有一个指向父节点的指针.
'''
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None

def GetNext(pnode):
    if pnode is None:
        return None 

    pnext = None
    if pnode.right is not None:
        pright = pnode.right
        while pright.left is not None:
            pright = pright.left
        pnext = pright
    elif pnode.parent is not None:
        pcurrent = pnode 
        pparent = pnode.parent
        # 可以与while句合并
        # if pcurrent == pparent.left:
        #     pnext = pparent
        while pparent is not None and pcurrent == pparent.right:
            pcurrent = pparent
            pparent = pcurrent.parent
        pnext = pparent 
    return pnext

# a = TreeNode("a")
# b,c = TreeNode("b"),TreeNode("c")
# d,e,f,g = TreeNode("d"),TreeNode("e"),TreeNode("f"),TreeNode("g") #构建完全二叉树
# a.left = b
# a.right = c
# b.left = d   
# b.right = e 
# b.parent = a
# c.left = f 
# c.right = g 
# c.parent = a
# d.parent = b
# e.parent = b
# f.parent = c
# g.parent = c
# print(GetNext(a).value) #节点有右子树
# print(GetNext(d).value) #节点没有右子树，且是父节点的左子节点
# print(GetNext(e).value) #节点没有右子树，且是父节点的右子节点
# print(GetNext(g)== None) #节点没有右子树，且是父节点的右子节点

    

'''
面试题10：斐波那契数列
题目一：求斐波那契数列的第n项。
写一个函数，输入n，求斐波那契（Fibonacci）数列的第n项。数列定义如下：
f(0)=0, f(1)=1, f(n)=f(n-1) + f(n-2)

题目二：青蛙跳台阶问题。
一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个n级台阶总共
有多少种跳法。
'''
def Fib(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    zero = 0
    one = 1
    for i in range(2, n+1):
        fibn = zero + one 
        zero = one
        one = fibn 
    return fibn


# for i in range(0, 12):
#     print(Fib(i))