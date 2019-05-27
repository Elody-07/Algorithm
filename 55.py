class TreeNode:
    def __init__(self, value):
        self.value = value 
        self.left = None 
        self.right = None

def IsBalanced(tree):
    if tree is None:
        isbalanced = True
        depth = 0
        return (isbalanced, depth)
    
    left_is, left_depth = IsBalanced(tree.left)
    right_is, right_depth = IsBalanced(tree.right)

    if (left_is and right_is):
        diff = left_depth - right_depth 
        if abs(diff) <= 1:
            return (True, 1 + max(left_depth, right_depth))
    
    return (False, 1+max(left_depth, right_depth))

def TreeDepth(tree):
    if tree is None:
        return 0
    
    left = TreeDepth(tree.left)
    right = TreeDepth(tree.right)

    return 1 + max(left, right)

a = TreeNode(1)
b = TreeNode(2)
c = TreeNode(3)
d = TreeNode(4)
e = TreeNode(5)
f = TreeNode(6)
g = TreeNode(7)
a.left = b 
a.right = c 
b.left = d  
b.right = e 
c.right = f 
e.left = g
    
print(TreeDepth(a))

