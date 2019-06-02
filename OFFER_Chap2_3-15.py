'''
----------------------------------------数组------------------------------------------------
'''

'''
面试题3：数组中重复的数字
题目一：找出数组中重复的数组。
在一个长度为 n 的数组中所有数字都在 0~n-1 范围内，找出数组中任意一个重复的数字。
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
题目二：不修改数组找出重复的数字。
在一个长度为 n+1 的数组里所有数字都在 1~n 的范围内，不修改输入的数组，找出数组中任意一个重复的数字。
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
题目：在一个二维数组中，每一行从左到右递增，每一列从上到下递增。请完成一个函数，输入这样的二维数组和一个整数，判断数组中是否含有该整数。
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
题目：实现一个函数，把字符串中的每个空格替换成 "%20"。如输入 "We are happy."，则输出 "We%20are%20happy."
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

def printLink(link):
    temp = link 
    while temp is not None:
        print(temp.value)
        temp = temp.next

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

# link1 = addToTail(None, 100)
# print(link1.value)
# print(link1.next)
# link1 = addToTail(link1, 50)
# link1 = addToTail(link1, 25)
# link1 = addToTail(link1, 12.5)
# link1 = addToTail(link1, 6.25)
# print("Add to tail:")
# printLink(link1)

# print(removeNode(None, 100))
# print(removeNode(link1, 0))
# link1, _= removeNode(link1, 100)
# link1, _ = removeNode(link1, 25)
# link1, _ = removeNode(link1, 6.25)
# print("Remove node:")
# printLink(link1)


'''
面试题6：从尾到头打印链表
题目：输入一个链表的头节点，从尾到头反过来打印出每个节点的值。
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
题目：输入某二叉树的前序遍历和中序遍历结果，重建该二叉树，假设输入的前序遍历和中序遍历的结果中都不包含重复的数字。如：输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，重建二叉树并输出它的头节点。二叉树节点定义如下：
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
题目：给定一棵二叉树和其中的一个节点，如何找出中序遍历序列的下一个节点？树中的节点除了有两个分别指向左、右子节点的指针，还有一个指向父节点的指针。
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
面试题9：用两个栈实现队列
题目：用两个栈实现队列。实现队列的另两个函数appendTail和deleteHead，分别完成在队列尾部插入节点和在队列头部删除节点的功能。
'''
def appendTail(stack1, stack2, value):
    stack1.append(value)

def deleteHead(stack1, stack2):
    if stack1 == [] and stack2 == []:
        raise RuntimeError('empty queue')

    if stack2 == 0:
        while len(stack1) > 0:
            stack2.append(stack1.pop())

    delete = stack2.pop()
    print(delete)
    return delete
    
# stack1, stack2 = [], []
# for i in range(5):
#     appendTail(stack1, stack2, i)
# print("stack1: ", stack1, "stack2: ", stack2)
# deleteHead(stack1, stack2)
# deleteHead(stack1, stack2)
# deleteHead(stack1, stack2)
# appendTail(stack1, stack2, 5)
# deleteHead(stack1, stack2)
# deleteHead(stack1, stack2)
# deleteHead(stack1, stack2)
# deleteHead(stack1, stack2)


'''
面试题10：斐波那契数列
题目一：求斐波那契数列的第n项。
写一个函数，输入n，求斐波那契（Fibonacci）数列的第n项。数列定义如下：f(0)=0, f(1)=1, f(n)=f(n-1) + f(n-2)

题目二：青蛙跳台阶问题。
一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个n级台阶总共有多少种跳法。
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


'''
面试题11：旋转数组的最小数字
题目：把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如数组[3,4,5,1,2]为[1,2,3,4,5]的一个旋转，该数组的最小值为1。
'''
def Rotate(arr):
    if arr == []:
        return None
    start = 0
    end = len(arr) - 1
    mid = start # important
    while arr[start] >= arr[end]:
        if end - start == 1:
            mid = end 
            break

        mid = (start + end) // 2
        # important
        if arr[start] == arr[mid] and arr[end] == arr[mid]:
            return RotateInOrder(arr, start, end)
        
        if arr[mid] >= arr[start]:
            start = mid
        if arr[mid] <= arr[end]:
            end = mid
    
    return arr[mid]

def RotateInOrder(arr, start, end):
    min = arr[start]
    for i in range(start, end+1):
        if arr[i] < min:
            min = arr[i]
    return min

# print(Rotate([3,4,5,1,2]))
# print(Rotate([3,4,4,1,1,2]))
# print(Rotate([1,2,3,4,5]))
# print(Rotate([1,0,1,1,1]))
# print(Rotate([1]))
# print(Rotate([]))
        



'''
面试题12：矩阵中的路径
题目：请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如在下面的3×4矩阵中包含一条字符串"bfce"的路径，但不包含"abfb"的路径，因为路径不能第二次进入字符b这个格子。
a  b  t  g
c  f  c  s
j  d  e  h
'''
def hasPath(arr, rows, cols, str):
    if arr == [] or rows < 1 or cols < 1 or str == '':
        return None
    
    visited = [0] * (rows * cols)
    pathLength = 0
    for row in range(0, rows):
        for col in range(0, cols):
            if find(arr, rows, cols, row, col, str, pathLength, visited):
                return True
    
    return False

def find(arr, rows, cols, row, col, str, pathLength, visited):
    if pathLength == len(str):
        return True
    
    flag = False
    if (row >= 0 and row < rows and 
        col >= 0 and col < cols and 
        arr[row*cols + col] == str[pathLength] and 
        visited[row*cols + col] == 0):

        pathLength += 1
        visited[row*cols + col] = 1
        flag = find(arr, rows, cols, row, col-1, str, pathLength, visited) or \
                  find(arr, rows, cols, row, col+1, str, pathLength, visited) or \
                  find(arr, rows, cols, row-1, col, str, pathLength, visited) or \
                  find(arr, rows, cols, row+1, col, str, pathLength, visited)
        if not flag:
            pathLength -= 1
            visited[row * cols + col] = 0
    
    return flag 

arr = ['a', 'b', 't', 'g',
       'c', 'f', 'c', 's',
       'j', 'd', 'e', 'h']
print(hasPath(arr, 3, 4, 'abfd'))
print(hasPath(arr, 3, 4, 'bfce'))
print(hasPath(arr, 3, 4, 'abfb'))
print(hasPath([], 0, 0, ''))
    